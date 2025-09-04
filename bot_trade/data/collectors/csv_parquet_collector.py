from __future__ import annotations

"""Collector reading local CSV/Parquet files.

Files are expected under ``data/ready`` by default. The loader performs light
validation: timezone normalisation to UTC, column reordering and duplicate
removal. Missing optional fields are filled with ``NaN``.
"""

from pathlib import Path

import pandas as pd

from .base import CollectorConfig, MarketCollector


class CSVParquetCollector(MarketCollector):
    def __init__(self, root: str | Path = "data/ready") -> None:
        self.root = Path(root)

    def _read_file(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported extension: {path.suffix}")
        return df

    def load(self, cfg: CollectorConfig) -> pd.DataFrame:
        pattern = f"{cfg.symbol}-{cfg.frame}"
        files = sorted(self.root.glob(f"**/{pattern}*.parquet"))
        if not files:
            files = sorted(self.root.glob(f"**/{pattern}*.csv"))
        if not files:
            raise FileNotFoundError(
                f"no files for symbol={cfg.symbol} frame={cfg.frame} in {self.root}"
            )
        dfs: list[pd.DataFrame] = []
        for fp in files:
            df = self._read_file(fp)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df.reset_index(inplace=True)
            df.rename(columns={"index": "datetime"}, inplace=True)
        df.sort_values("datetime", inplace=True)
        df.drop_duplicates(subset=["datetime"], inplace=True)
        if cfg.start:
            df = df[df["datetime"] >= pd.to_datetime(cfg.start, utc=True)]
        if cfg.end:
            df = df[df["datetime"] <= pd.to_datetime(cfg.end, utc=True)]
        df.set_index("datetime", inplace=True)
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = float("nan")
        for opt in ["spread_bp", "best_bid", "best_ask", "depth_top"]:
            if opt not in df.columns:
                df[opt] = float("nan")
        return df[required + ["spread_bp", "best_bid", "best_ask", "depth_top"]]
