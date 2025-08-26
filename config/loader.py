"""
config/loader.py — Unified, robust data loader for Boot Trade

Goals:
- Discover files per (symbol, frame)
- Read Feather/Parquet/CSV quickly with downcasting to float32/int32
- Unify timestamps (s/ms/us/ns) → UTC `datetime` column
- Ensure required columns (open/high/low/close/volume, symbol, frame, datetime)
- Preserve ALL numeric features if present (do **not** limit to 6 cols)
- Optional feature engineering hook (add_strategy_features) — off by default to avoid duplication with env
- Light validation when safe=True

Designed to be used from `train_rl.py` and `env_trading.py` without import cycles.
"""
from __future__ import annotations
import os
import re
import glob
import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict

import numpy as np
import pandas as pd
# module-level logger: avoid propagating noisy logs from workers on Windows
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())


# Optional hook (import lazily inside functions to avoid import cycles)
AddFeaturesFn = Optional[Callable[[pd.DataFrame], pd.DataFrame]]

REQUIRED_BASE = ["open", "high", "low", "close", "volume"]


# ============================
# Helpers
# ============================

def _infer_unit_from_ts(sample: float) -> str:
    """Infer pandas unit for pd.to_datetime from a numeric timestamp sample."""
    if not np.isfinite(sample):
        return "s"
    # Approximate by magnitude
    mag = math.log10(abs(sample) + 1e-9)
    if mag < 10.5:
        return "s"   # seconds (<= ~year 2286)
    if mag < 13.5:
        return "ms"  # milliseconds
    if mag < 16.5:
        return "us"  # microseconds
    return "ns"      # nanoseconds


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Already has datetime?
    if "datetime" in df.columns:
        # coerce & set UTC
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        return df
    # 2) common epoch cols
    for c in ("timestamp_ms", "ts_ms", "time_ms"):
        if c in df.columns:
            df["datetime"] = pd.to_datetime(df[c], unit="ms", utc=True, errors="coerce")
            return df
    for c in ("timestamp", "ts", "time"):
        if c in df.columns:
            # Try to infer unit from a sample
            try:
                sample = float(df[c].iloc[0]) if len(df) else 0.0
            except Exception:
                sample = 0.0
            unit = _infer_unit_from_ts(sample)
            df["datetime"] = pd.to_datetime(df[c], unit=unit, utc=True, errors="coerce")
            return df
    # 3) fallback: synthetic monotonic datetime (seconds)
    df["datetime"] = pd.to_datetime(range(len(df)), unit="s", utc=True)
    return df


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # If high/low are missing, approximate conservatively from available columns
    if "open" in df.columns and "close" in df.columns:
        if "high" not in df.columns:
            df["high"] = df[["open", "close"]].max(axis=1)
        if "low" not in df.columns:
            df["low"] = df[["open", "close"]].min(axis=1)
    # Ensure volume exists
    if "volume" not in df.columns:
        # Some datasets use 'vol' or 'quote_volume'
        for alt in ("vol", "volume_usd", "quote_volume", "base_volume"):
            if alt in df.columns:
                df["volume"] = df[alt]
                break
        else:
            df["volume"] = 0.0
    return df


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def _parse_symbol_frame_from_name(path: str) -> Dict[str, Optional[str]]:
    name = os.path.basename(path)
    # Try patterns like: BTCUSDT-1m-2023.feather or 1m_BTCUSDT_*.parquet
    m = re.search(r"(?P<sym>[A-Z0-9]{3,20})[-_](?P<frame>[0-9]+[smhdw])", name)
    sym = m.group("sym") if m else None
    frame = m.group("frame") if m else None
    # Fallbacks
    if sym is None:
        m2 = re.search(r"(?P<sym>[A-Z0-9]{3,20})", name)
        sym = m2.group("sym") if m2 else None
    if frame is None:
        m3 = re.search(r"(?P<frame>[0-9]+[smhdw])", name)
        frame = m3.group("frame") if m3 else None
    return {"symbol": sym, "frame": frame}


# ============================
# Public API
# ============================

@dataclass
class LoadOptions:
    numeric_only: bool = True
    add_features: bool = False  # compute indicators via strategy_features
    safe: bool = False
    expect_symbol: Optional[str] = None
    expect_frame: Optional[str] = None


def discover_files(frame: str, symbol: str, data_root: str = "data") -> List[str]:
    """Return sorted list of files under data_root/<frame>/ matching symbol.
    Prefer Feather; fall back to Parquet then CSV.
    """
    patt_base = os.path.join(data_root, frame)
    cands = []
    for ext in ("feather", "parquet", "csv"):
        pattern1 = os.path.join(patt_base, f"{symbol}-{frame}-*.{ext}")
        pattern2 = os.path.join(patt_base, f"*{symbol}*.{ext}")
        cands += glob.glob(pattern1)
        cands += glob.glob(pattern2)
        if cands:
            break  # prefer the first extension found
    files = sorted(set(cands))
    if not files:
        raise FileNotFoundError(f"No data files found for {symbol=} {frame=} in {patt_base}")
    return files


def read_one(path: str, opts: LoadOptions = LoadOptions()) -> pd.DataFrame:
    """
    Read a single file (feather/parquet/csv) and standardize columns.
    - Preserves all numeric features (numeric_only=True: drops non-numeric like strings)
    - Adds `datetime` (UTC), `symbol`, `frame` if missing
    - Downcasts to float32/int32 for memory efficiency
    - Optionally calls `config.strategy_features.add_strategy_features`
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".feather":
        try:
            import pyarrow.feather as paw
            tbl = paw.read_table(path)
            df = tbl.to_pandas(types_mapper=None)
        except ImportError as e:
            raise ImportError(
                "Feather requires 'pyarrow'. ثبتها: pip install pyarrow "
                "أو conda install -c conda-forge pyarrow"
            ) from e
        except Exception:
            # fallback to pandas API
            df = pd.read_feather(path)
    elif ext == ".parquet":
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except ImportError:
            try:
                df = pd.read_parquet(path, engine="fastparquet")
            except Exception as e:
                raise ImportError(
                    "Parquet requires 'pyarrow' أو 'fastparquet'. ثبّت أحدهما."
                ) from e
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # ✅ أولاً: وحّد/كوّن datetime قبل أي فلترة رقمية
    df = _ensure_datetime(df)

    # ثم ثبّت symbol/frame
    meta = _parse_symbol_frame_from_name(path)
    symbol = opts.expect_symbol or meta.get("symbol") or (df["symbol"].iloc[0] if "symbol" in df.columns and len(df)>0 else None)
    frame = opts.expect_frame or meta.get("frame") or (df["frame"].iloc[0] if "frame" in df.columns and len(df)>0 else None)
    if symbol is None:
        symbol = "UNKNOWN"
    if frame is None:
        frame = "1m"
    df["symbol"] = str(symbol).upper()
    df["frame"] = str(frame)

    # ✅ ثم نطبّق numeric_only لكن مع الحفاظ على meta الضرورية
    if opts.numeric_only:
        keep_meta = [c for c in ("datetime","symbol","frame") if c in df.columns]
        num = df.select_dtypes(include=[np.number]).copy()
        for c in keep_meta:
            if c not in num.columns:
                num[c] = df[c]
        df = num

    # ضمان OHLCV
    df = _ensure_ohlcv(df)

    # downcast
    df = _downcast_numeric(df)

    # ميزات اختيارية
    if opts.add_features:
        try:
            from config.strategy_features import add_strategy_features
            df = add_strategy_features(df)
        except Exception as e:
            _logger.warning("[features] add_strategy_features failed: %s", e)

    # فحوصات أمان اختيارية
    if opts.safe:
        problems: Dict[str, str] = {}
        for col in REQUIRED_BASE:
            if col not in df.columns:
                problems[col] = "missing"
        if "datetime" not in df.columns or not df["datetime"].notna().all():
            problems["datetime"] = "missing or NaT present"
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty and num_df.isna().any().any():
            problems["NaN"] = "numeric NaN present"
        if problems:
            _logger.warning("[SAFE] Potential issues in %s: %s", os.path.basename(path), problems)

    if df.empty:
        raise ValueError(f"Loaded empty dataframe from {path}")

    return df



def load_dataset(frame: str, symbol: str, data_root: str = "data", opts: LoadOptions = LoadOptions()) -> pd.DataFrame:
    """Load and concatenate all files for (frame, symbol), sorted by datetime, reset index.
    Guarantees columns: datetime(UTC), symbol, frame, OHLCV, and preserves all numeric features found.
    """
    files = discover_files(frame=frame, symbol=symbol, data_root=data_root)
    parts: List[pd.DataFrame] = []
    for fp in files:
        part = read_one(fp, opts=LoadOptions(**{**opts.__dict__, "expect_symbol": symbol, "expect_frame": frame}))
        parts.append(part)
    if not parts:
        raise RuntimeError("No parts loaded (unexpected)")
    df = pd.concat(parts, axis=0, ignore_index=True)
    # Ensure monotonic by datetime, then by an available idx if ties
    if "datetime" in df.columns:
        df.sort_values(["datetime"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def to_env_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view with numeric features only (float32/int32) and required meta cols removed
    (env builds obs from numeric cols and uses symbol/datetime separately).
    """
    num = df.select_dtypes(include=[np.number]).copy()
    # Keep meta (symbol/frame/datetime) alongside numeric for env indexing if needed
    if "symbol" not in num.columns:
        num["symbol"] = 0  # placeholder numeric so selection keeps schema
    if "frame" not in num.columns:
        num["frame"] = 0
    # NOTE: env_trading will set MultiIndex([symbol, datetime]) internally
    return num


# Convenience runner for quick manual tests
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frame", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--features", action="store_true")
    p.add_argument("--safe", action="store_true")
    args = p.parse_args()

    opts = LoadOptions(numeric_only=True, add_features=bool(args.features), safe=bool(args.safe),
                       expect_symbol=args.symbol, expect_frame=args.frame)
    df = load_dataset(args.frame, args.symbol, data_root=args.data_root, opts=opts)
    print(df.head(3))
    print("shape:", df.shape)
    print("cols:", list(df.columns)[:20])
