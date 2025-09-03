from __future__ import annotations

"""Lightweight data store inspection."""

import argparse
from pathlib import Path

from bot_trade.data.loaders import load_with_stats
from bot_trade.tools.encoding import force_utf8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect data store")
    ap.add_argument("--root", default="data_store", help="Data storage root")
    ns = ap.parse_args(argv)

    root = Path(ns.root).resolve()
    gaps = dups = files = 0
    if root.exists():
        for p in root.rglob("*.parquet"):
            frame = p.parent.name
            try:
                _, g, d = load_with_stats(p, frame)
                gaps += g
                dups += d
                files += 1
            except Exception:
                continue
    missing = 0 if files else 1
    print(
        f"[DATA] root={root} gaps={gaps} dups={dups} missing={missing}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
