from __future__ import annotations

"""Parquet storage for OHLCV datasets with strict schema enforcement.

Schema:
    ts      int64 nanoseconds since epoch (UTC)
    open    float32
    high    float32
    low     float32
    close   float32
    volume  float32
    symbol  string
    frame   string

All timestamps must be in UTC and aligned to a unified trading calendar.
"""

import os
from pathlib import Path
import pandas as pd

from .validators import enforce_schema, enforce_monotonic_ts


def write_parquet_atomic(path: str | Path, df: pd.DataFrame) -> Path:
    """Atomically write ``df`` to ``path`` enforcing schema and monotonic ts."""

    df = enforce_schema(df)
    enforce_monotonic_ts(df)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    return dest


def read_parquet_strict(path: str | Path) -> pd.DataFrame:
    """Read Parquet file from ``path`` applying schema and timestamp checks."""

    df = pd.read_parquet(Path(path))
    df = enforce_schema(df)
    enforce_monotonic_ts(df)
    return df
