from __future__ import annotations

"""Lightweight data store inspection."""

import argparse
from pathlib import Path

from bot_trade.data.store_parquet import read_parquet_strict
from bot_trade.data.validators import detect_gaps, detect_duplicates
from bot_trade.tools.force_utf8 import force_utf8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect data store")
    ap.add_argument("--root", default="data_store", help="Data storage root")
    ns = ap.parse_args(argv)

    root = Path(ns.root).resolve()
    gaps = dups = 0
    files = 0
    frames: set[str] = set()
    if root.exists():
        for p in root.rglob("*.parquet"):
            frame = p.parent.name
            try:
                df = read_parquet_strict(p)
                gaps += detect_gaps(df, frame)
                dups += detect_duplicates(df)
                frames.add(frame)
                files += 1
            except Exception:
                continue
    print(
        f"[DATA_DOCTOR] root={root} files={files} frames={len(frames)} gaps={gaps} dups={dups}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
