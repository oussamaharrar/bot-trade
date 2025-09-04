"""Live dry-run runner consuming live prices and executing paper trades."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import yaml

from bot_trade.data.live_feed import LiveFeed
from bot_trade.tools.atomic_io import append_jsonl, write_json, write_text
from bot_trade.tools.force_utf8 import force_utf8


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(description="Live dry-run")
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--gateway", default="paper")
    ap.add_argument("--model", help="Path to model zip")
    ap.add_argument(
        "--model-optional",
        action="store_true",
        help="Allow missing model and fall back to random actions",
    )
    ap.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Run duration in seconds",
    )
    ap.add_argument("--config", default="config/live_dry_run.yaml")
    args = ap.parse_args()
    args.duration = max(int(args.duration), 1)

    cfg = _load_config(args.config)
    ecfg = cfg[args.exchange]
    feed = LiveFeed(ecfg.get("ws"), ecfg["rest"] + ecfg["price_path"], interval=1.0)

    class HoldPolicy:
        def action(self, _: float) -> str:
            return "hold"

    class RandomPolicy:
        def __init__(self) -> None:
            self.rng = random.Random()

        def action(self, _: float) -> str:
            return self.rng.choice(["buy", "sell", "hold"])

    policy: object
    if args.model and Path(args.model).exists():
        policy = HoldPolicy()  # placeholder for real model policy
    elif not args.model_optional:
        raise SystemExit("model required")
    else:
        print("[WARN] model missing, falling back to RandomPolicy")
        policy = RandomPolicy()

    run_dir = Path("results") / args.symbol / args.frame / str(int(time.time()))
    metrics = []

    def on_tick(price: float) -> None:
        action = policy.action(price)
        metrics.append({"ts": time.time(), "price": price, "action": action})
        print(f"[LIVE] tick={price} action={action}")

    feed.stream(args.symbol, on_tick)
    start = time.time()
    try:
        while time.time() - start < args.duration:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {"ticks": len(metrics)}
        write_json(run_dir / "summary.json", summary)
        for m in metrics:
            append_jsonl(run_dir / "risk_flags.jsonl", m)
        # metrics.csv
        csv_path = run_dir / "metrics.csv"
        if metrics:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=metrics[0].keys())
                w.writeheader()
                w.writerows(metrics)
        else:
            write_text(csv_path, "")


if __name__ == "__main__":
    main()
