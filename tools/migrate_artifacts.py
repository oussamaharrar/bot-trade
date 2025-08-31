#!/usr/bin/env python
"""Migrate legacy model artifacts to new dual-archive layout."""
from __future__ import annotations

import argparse
import csv
import shutil
import datetime as dt
from pathlib import Path

from bot_trade.config.rl_paths import get_root


def migrate(symbol: str, frame: str) -> None:
    root = get_root()
    agents = root / "agents" / symbol.upper() / str(frame)
    archive = agents / "archive"
    archive_best = agents / "archive_best"
    archive_best.mkdir(parents=True, exist_ok=True)
    index = archive_best / "index.csv"
    idx_exists = index.exists()
    idx_fh = index.open("a", encoding="utf-8", newline="")
    writer = csv.writer(idx_fh)
    if not idx_exists:
        writer.writerow(["ts", "run_id", "metric", "model_path", "vecnorm_path"])
    moved: list[tuple[str, str]] = []
    for file in archive.glob("deep_rl_best-*.zip"):
        dst = archive_best / file.name
        shutil.move(str(file), dst)
        writer.writerow([dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S"), "legacy", "", str(dst), ""])
        moved.append((str(file), str(dst)))
    idx_fh.close()
    if moved:
        report_dir = root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report = report_dir / f"migration_{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
        with report.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["src", "dst"])
            w.writerows(moved)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    args = ap.parse_args()
    migrate(args.symbol, args.frame)


if __name__ == "__main__":
    main()
