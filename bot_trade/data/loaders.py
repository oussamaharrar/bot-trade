from __future__ import annotations

"""Convenience loaders wrapping Parquet store and validators."""

from pathlib import Path
import pandas as pd

from .store_parquet import load_ohlcv
from .validators import gap_dup_stats


def load_with_stats(path: Path, frame: str) -> tuple[pd.DataFrame, int, int]:
    """Load dataset and return (df, gaps, dups)."""

    df = load_ohlcv(path)
    gaps, dups = gap_dup_stats(df, frame)
    return df, gaps, dups

