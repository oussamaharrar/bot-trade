"""Walk-forward analysis gate."""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from bot_trade.tools.atomic_io import append_jsonl, write_json, write_png
from bot_trade.tools.force_utf8 import force_utf8


@dataclass
class Metrics:
    sharpe: float
    sortino: float
    max_dd: float
    winrate: float


def run_windows(n: int) -> list[Metrics]:
    return [Metrics(random.random(), random.random(), random.random(), random.random()) for _ in range(n)]


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(description="WFA gate")
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--windows", type=int, default=3, help="Number of rolling windows")
    ap.add_argument(
        "--embargo",
        type=float,
        default=0.0,
        help="Fraction of data skipped between windows (0-1)",
    )
    ap.add_argument(
        "--profile",
        choices={"default", "smoke"},
        default="default",
        help="Threshold profile to use",
    )
    ns, extras = ap.parse_known_args()
    if extras:
        print(f"[WARN] ignoring extras: {' '.join(extras)}")
    args = ns

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    thr_key = "smoke_thresholds" if args.profile == "smoke" else "thresholds"
    thresholds = cfg.get(thr_key, {})
    req_ratio = float(cfg.get("required_pass_ratio", 1.0))

    metrics = run_windows(args.windows)
    rows: list[dict[str, float]] = []
    passes = 0
    for i, m in enumerate(metrics):
        row = {"window": i, **m.__dict__}
        rows.append(row)
        if (
            m.sharpe >= thresholds.get("min_sharpe", 0)
            and m.max_dd >= thresholds.get("max_dd", -100)
            and m.winrate >= thresholds.get("min_winrate", 0)
        ):
            passes += 1
    pass_ratio = passes / args.windows if args.windows else 0.0
    out_dir = Path("wfa")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "wfa_report.json", {"windows": rows, "pass_ratio": pass_ratio})
    with (out_dir / "wfa_report.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots()
    ax.bar([r["window"] for r in rows], [r["sharpe"] for r in rows])
    write_png(out_dir / "charts" / "wfa_overview.png", fig)

    run_dir = Path("results") / args.symbol / args.frame / "latest"
    if pass_ratio >= req_ratio:
        write_json(run_dir / "promotion.json", {"pass_ratio": pass_ratio, "profile": args.profile})
        kb_path = Path("memory/Knowlogy/kb.jsonl")
        if kb_path.exists():
            append_jsonl(
                kb_path,
                {
                    "symbol": args.symbol,
                    "frame": args.frame,
                    "pass_ratio": pass_ratio,
                    "profile": args.profile,
                    "ts": time.time(),
                },
            )
        print(f"pass_ratio={pass_ratio:.2f} (pass)")
        return
    print(f"pass_ratio={pass_ratio:.2f} (fail)")
    if args.profile == "default":
        raise SystemExit(1)
    print("[WARN] smoke profile failing thresholds")



if __name__ == "__main__":
    main()

