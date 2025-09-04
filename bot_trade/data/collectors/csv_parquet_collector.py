from __future__ import annotations

"""Collector reading local CSV/Parquet files.

Files are expected under ``data/ready`` by default. The loader performs light
validation: timezone normalisation to UTC, column reordering and duplicate
removal. Missing optional fields are filled with ``NaN``.
"""

from pathlib import Path

import pandas as pd

from .base import CollectorConfig, MarketCollector
from bot_trade.data.discovery import discover_raw_series, assert_non_empty


class CSVParquetCollector(MarketCollector):
    def _read_file(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported extension: {path.suffix}")

    def load(self, cfg: CollectorConfig) -> pd.DataFrame:
        raw_dir = cfg.raw_dir or "data/ready"
        files = discover_raw_series(raw_dir, cfg.symbol, cfg.frame)
        assert_non_empty(files, cfg.symbol, cfg.frame, raw_dir)
        dfs = [self._read_file(Path(fp)) for fp in files]
        df = pd.concat(dfs, ignore_index=True)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df.set_index("datetime", inplace=True)
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        if cfg.start:
            df = df[df.index >= pd.to_datetime(cfg.start, utc=True)]
        if cfg.end:
            df = df[df.index <= pd.to_datetime(cfg.end, utc=True)]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = float("nan")
        for opt in ["spread_bp", "best_bid", "best_ask", "depth_top"]:
            if opt not in df.columns:
                df[opt] = float("nan")
        df.ffill(limit=5, inplace=True)
        while len(df) and df.iloc[0].isna().any():
            df = df.iloc[1:]
        while len(df) and df.iloc[-1].isna().any():
            df = df.iloc[:-1]
        return df[required + ["spread_bp", "best_bid", "best_ask", "depth_top"]]
