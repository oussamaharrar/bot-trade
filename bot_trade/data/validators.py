from __future__ import annotations

"""Data validation helpers for OHLCV datasets."""

from typing import Tuple
import pandas as pd

REQUIRED_COLS = [
    "ts",
    "symbol",
    "frame",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
]


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to canonical dtypes and order."""

    d = df.copy()
    if "ts" in d.columns:
        d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce").view("int64")
    d["symbol"] = d.get("symbol", "").astype("string")
    d["frame"] = d.get("frame", "").astype("string")
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d.get(col, 0.0), downcast="float")
    if "trade_count" in d.columns:
        d["trade_count"] = pd.to_numeric(d["trade_count"], downcast="integer").astype("Int32")
    else:
        d["trade_count"] = pd.Series([pd.NA] * len(d), dtype="Int32")
    return d[REQUIRED_COLS]


def gap_dup_stats(df: pd.DataFrame, frame: str) -> Tuple[int, int]:
    """Return (gaps, dups) counts for ``df`` given ``frame`` frequency."""

    freq = FRAME_TO_PANDAS.get(frame, frame)
    ts = pd.to_datetime(df["ts"], unit="ns", utc=True)
    expected = pd.date_range(ts.min(), ts.max(), freq=freq)
    dups = int(ts.duplicated().sum())
    gaps = int(len(expected) - ts.nunique())
    return gaps, dups


FRAME_TO_PANDAS = {
    "1m": "1min",
    "1h": "1H",
    "1d": "1D",
}

