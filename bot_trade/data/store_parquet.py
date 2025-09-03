from __future__ import annotations

"""Parquet storage helpers for OHLCV bars."""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .validators import enforce_schema


def write_ohlcv(df: pd.DataFrame, dest: Path) -> Path:
    """Write ``df`` to ``dest`` ensuring canonical schema.

    DataFrame columns are coerced to the required dtypes via
    :func:`enforce_schema`.  The write is atomic (``.tmp`` then rename).
    """

    df = enforce_schema(df)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    return dest


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load OHLCV bars from ``path`` and enforce schema."""

    df = pd.read_parquet(Path(path))
    return enforce_schema(df)

