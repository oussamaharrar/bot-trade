from __future__ import annotations

"""Atomic Parquet read/write stubs."""

from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

