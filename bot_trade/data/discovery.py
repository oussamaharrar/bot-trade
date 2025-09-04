from __future__ import annotations

"""Raw data discovery helpers honoring ``--raw-dir``."""

from pathlib import Path
from typing import List

VALID_EXT = (".parquet", ".csv")


def discover_raw_series(raw_dir: str, symbol: str, frame: str) -> List[str]:
    """Return candidate OHLCV files for ``symbol``/``frame`` under ``raw_dir``.

    The search is tolerant to a few common layouts::

        <raw_dir>/<symbol>/<frame>/*
        <raw_dir>/<symbol>_<frame>/*
        <raw_dir>/<symbol>_<frame>.*
        <raw_dir>/<symbol>/<symbol>_<frame>.*
    """

    d = Path(raw_dir)
    patterns = [
        d / symbol / frame / "*",
        d / f"{symbol}_{frame}" / "*",
        d / f"{symbol}_{frame}.*",
        d / symbol / f"{symbol}_{frame}.*",
    ]
    files = []
    for p in patterns:
        if "*" in str(p):
            files += [x for x in p.parent.glob(p.name) if x.suffix.lower() in VALID_EXT]
        elif p.exists() and p.suffix.lower() in VALID_EXT:
            files.append(p)
    files = sorted(set(files), key=lambda x: (x.name, str(x)))
    return [str(x) for x in files]


def assert_non_empty(files: List[str], symbol: str, frame: str, raw_dir: str) -> None:
    """Abort with a friendly message if ``files`` is empty."""

    if not files:
        print(
            f"[DATA] no files for {symbol} {frame} in {raw_dir}. "
            f"Try {raw_dir}/{symbol}/{frame}/*.parquet or {raw_dir}/{symbol}_{frame}.parquet"
        )
        raise SystemExit(2)
