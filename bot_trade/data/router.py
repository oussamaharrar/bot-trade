from __future__ import annotations

"""Data routing utilities for raw and live market sources."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .collectors.base import CollectorConfig
from .collectors.csv_parquet_collector import CSVParquetCollector
from .collectors.ccxt_rest_collector import CCXTRestCollector
from .collectors.ccxt_ws_collector import CCXTWSCollector


TF_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"}


def to_ccxt_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    if len(sym) > 4:
        return f"{sym[:-4]}/{sym[-4:]}"
    return sym


@dataclass
class DataRouter:
    mode: str = "raw"
    source: str = "csvparquet"
    raw_dir: str = "data/ready"
    exchange: str | None = None
    cache_dir: str = "data/cache"

    def _collector(self) -> object:
        if self.mode == "raw":
            return CSVParquetCollector(self.raw_dir)
        if self.source == "ccxt-ws":
            return CCXTWSCollector(self.exchange or "binance")
        return CCXTRestCollector(self.exchange or "binance")

    def load(
        self,
        symbol: str,
        frame: str,
        start: str | None = None,
        end: str | None = None,
        ccxt_symbol: str | None = None,
        signals_spec: str | None = None,
    ) -> pd.DataFrame:
        mapped = TF_MAP.get(frame)
        if mapped is None:
            print(f"[DATA] unsupported timeframe: {frame}")
            raise SystemExit(3)
        coll = self._collector()
        sym = symbol
        if self.mode != "raw":
            sym = ccxt_symbol or to_ccxt_symbol(symbol)
        cfg = CollectorConfig(
            symbol=sym,
            frame=mapped,
            start=start,
            end=end,
            exchange=self.exchange,
            cache_dir=self.cache_dir,
        )
        df = coll.load(cfg)
        if df is None or len(df) == 0:
            ex = self.exchange or "csv"
            csym = sym
            print(
                f"[DATA] no OHLCV fetched for {ex} {csym} {mapped}; adjust --start/--end or check network"
            )
            raise SystemExit(2)
        return df

