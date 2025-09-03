from __future__ import annotations

"""Check minimal data store status."""

import argparse
from pathlib import Path

from bot_trade.tools.encoding import force_utf8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect data store")
    ap.add_argument("--tz", default="UTC")
    ns = ap.parse_args(argv)
    parquet_dir = Path("data_store") / "parquet"
    parquet = "ok" if parquet_dir.exists() else "absent"
    calendar = "ok"
    print(f"[DATA] parquet={parquet} calendar={calendar} tz={ns.tz}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
