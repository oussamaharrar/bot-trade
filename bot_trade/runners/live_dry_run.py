from __future__ import annotations
"""Live dry-run runner consuming live prices and executing paper trades."""

import argparse
import random
import time
from pathlib import Path
from typing import Callable

import yaml

from bot_trade.data.live_feed import LiveFeed
from bot_trade.gateways.paper import PaperGateway
from bot_trade.tools.atomic_io import append_jsonl, write_json
from bot_trade.tools.force_utf8 import force_utf8


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(description="Live dry-run")
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--gateway", default="paper")
    ap.add_argument("--model")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--config", default="config/live_dry_run.yaml")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    ecfg = cfg[args.exchange]
    feed = LiveFeed(ecfg.get("ws"), ecfg["rest"] + ecfg["price_path"], interval=1.0)
    gw = PaperGateway(args.symbol)

    run_dir = Path("results") / args.symbol / args.frame / str(int(time.time()))
    metrics = []

    def on_tick(price: float) -> None:
        action = random.choice(["buy", "sell", "hold"])
        metrics.append({"ts": time.time(), "price": price, "action": action})
        print(f"[LIVE] tick={price} action={action}")

    feed.stream(args.symbol, on_tick)
    start = time.time()
    try:
        while time.time() - start < args.duration:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    summary = {"ticks": len(metrics)}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "summary.json", summary)
    for m in metrics:
        append_jsonl(run_dir / "risk_flags.jsonl", m)
    write_json(run_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
