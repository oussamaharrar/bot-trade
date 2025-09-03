from __future__ import annotations

"""Data validation helpers for OHLCV datasets.

Columns: ts (int nanoseconds UTC), open/high/low/close/volume (float32),
symbol (string), frame (string). A unified trading calendar is expected per
symbol/frame combination.
"""

import pandas as pd

REQUIRED_COLS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "frame",
]

FRAME_TO_PANDAS = {
    "1m": "1min",
    "1h": "1H",
    "1d": "1D",
}


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with columns coerced to canonical dtypes and order."""

    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce").view("int64")
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d.get(col, 0.0), downcast="float")
    d["symbol"] = d.get("symbol", "").astype("string")
    d["frame"] = d.get("frame", "").astype("string")
    return d[REQUIRED_COLS]


def detect_gaps(df: pd.DataFrame, frame: str) -> int:
    """Return count of missing periods based on ``frame`` frequency."""

    ts = pd.to_datetime(df["ts"], unit="ns", utc=True)
    if ts.empty:
        return 0
    freq = FRAME_TO_PANDAS.get(frame, frame)
    expected = pd.date_range(ts.min(), ts.max(), freq=freq)
    return int(len(expected) - ts.nunique())


def detect_duplicates(df: pd.DataFrame) -> int:
    """Return number of duplicate timestamps."""

    ts = pd.to_datetime(df["ts"], unit="ns", utc=True)
    return int(ts.duplicated().sum())


def enforce_monotonic_ts(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if timestamps are not strictly increasing."""

    ts = pd.to_datetime(df["ts"], unit="ns", utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps not monotonic increasing")
