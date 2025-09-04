from __future__ import annotations

"""Collector fetching data via ccxt if available."""

from pathlib import Path

import pandas as pd

from .base import CollectorConfig, MarketCollector


class CCXTRestCollector(MarketCollector):
    def __init__(self, exchange: str, root: str | Path = "data/cache") -> None:
        try:
            import ccxt  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            print("[DATA] ccxt not installed; pip install ccxt or use --data-mode raw")
            raise SystemExit(1)
        self.ccxt = ccxt
        self.exchange_id = exchange
        self.root = Path(root)

    def _exchange(self):
        return getattr(self.ccxt, self.exchange_id)()

    def load(self, cfg: CollectorConfig) -> pd.DataFrame:
        ex = self._exchange()
        limit = 1000
        since = int(pd.Timestamp(cfg.start or 0, tz="UTC").timestamp() * 1000)
        all_rows = []
        while True:
            chunk = ex.fetch_ohlcv(cfg.symbol, timeframe=cfg.frame, since=since, limit=limit)
            if not chunk:
                break
            all_rows.extend(chunk)
            since = chunk[-1][0] + 1
            if len(chunk) < limit:
                break
        if not all_rows:
            print(f"[DATA] no OHLCV fetched for {self.exchange_id} {cfg.symbol} {cfg.frame}; adjust --start/--end or check network")
            raise SystemExit(2)
        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        for opt in ["spread_bp", "best_bid", "best_ask", "depth_top"]:
            df[opt] = float("nan")
        cache_dir = Path(cfg.cache_dir or self.root) / self.exchange_id / cfg.symbol / cfg.frame
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "data.parquet"
            df.to_parquet(cache_file)
        except Exception:
            pass
        return df[["open", "high", "low", "close", "volume", "spread_bp", "best_bid", "best_ask", "depth_top"]]
