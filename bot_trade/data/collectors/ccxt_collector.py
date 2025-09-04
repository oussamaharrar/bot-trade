from __future__ import annotations

"""Collector fetching data via ccxt if available."""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import MarketCollector


class CCXTCollector(MarketCollector):
    def __init__(self, exchange: str, root: str | Path = "data/cache") -> None:
        try:
            import ccxt  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            print("[DATA] ccxt not installed; use CSV/Parquet collector")
            raise SystemExit(1)
        self.ccxt = ccxt
        self.exchange_id = exchange
        self.root = Path(root)

    def _exchange(self):
        return getattr(self.ccxt, self.exchange_id)()

    def load(
        self,
        symbol: str,
        frame: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        ex = self._exchange()
        limit = 1000
        since = int(pd.Timestamp(start or 0, tz="UTC").timestamp() * 1000)
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=frame, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        for opt in ["spread_bp", "best_bid", "best_ask", "depth_top"]:
            df[opt] = float("nan")
        return df[["open", "high", "low", "close", "volume", "spread_bp", "best_bid", "best_ask", "depth_top"]]
