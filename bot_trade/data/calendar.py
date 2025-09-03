from __future__ import annotations

"""Unified trading calendars per timeframe."""

import pandas as pd

from .validators import FRAME_TO_PANDAS


def calendar_index(start_ts: int, end_ts: int, frame: str) -> pd.DatetimeIndex:
    """Return a DatetimeIndex between ``start_ts`` and ``end_ts`` for ``frame``."""

    freq = FRAME_TO_PANDAS.get(frame, frame)
    return pd.date_range(
        pd.to_datetime(int(start_ts), unit="ns", utc=True),
        pd.to_datetime(int(end_ts), unit="ns", utc=True),
        freq=freq,
    )

