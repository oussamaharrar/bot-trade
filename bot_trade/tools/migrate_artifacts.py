from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import List

from bot_trade.config.rl_paths import get_root


def migrate() -> Path:
    root = get_root()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"migration_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["old", "new"])
    return out


def main(argv: List[str] | None = None) -> int:
    argparse.ArgumentParser("migrate artifacts").parse_args(argv)
    path = migrate()
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
