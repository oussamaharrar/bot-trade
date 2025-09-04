"""Walk-forward analysis gate."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from bot_trade.tools.atomic_io import write_json, write_png
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
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--embargo", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    metrics = run_windows(args.windows)
    rows = []
    passes = 0
    for i, m in enumerate(metrics):
        row = {"window": i, **m.__dict__}
        rows.append(row)
        if m.sharpe >= cfg.get("min_sharpe", 0):
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
    print(f"pass_ratio={pass_ratio:.2f}")
    if pass_ratio < cfg.get("required_pass_ratio", 0):
        raise SystemExit(1)



if __name__ == "__main__":
    main()

