#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ready-to-run extractor that outputs Feather files compatible with TradingEnv.

Key behavior:
- Reads Binance Kline CSVs (12 cols) under --raw-dir/<frame> (e.g., data/1s/*.csv)
- Normalizes epoch units: detects seconds → converts to milliseconds
- Enforces schema:
    timestamp(int64), datetime(datetime64[ns]), symbol(string), frame(string),
    open, high, low, close, volume, quote_volume, num_trades, taker_buy_base, taker_buy_quote (float32)
- Parallel CPU (Windows-safe): ≤ 60 workers, imap_unordered with smart chunksize
- For ALL frames (1s,1m,3m,5m): MERGE first → one Feather per SYMBOL
- For 1s ONLY (when --split-1s year): AFTER merge, ALSO split into per-year Feather parts per SYMBOL

Example:
    python ./prepare_data_ready_final.py --raw-dir data --out-dir data_ready --frames 1s 1m 3m 5m --split-1s year
"""

from __future__ import annotations
import glob
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
CSV_COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_volume","num_trades",
    "taker_buy_base","taker_buy_quote","ignore",
]
KEEP_COLS = [
    "open_time","open","high","low","close","volume",
    "quote_volume","num_trades","taker_buy_base","taker_buy_quote",
]
FINAL_COLS = [
    "timestamp","datetime","symbol","frame",
    "open","high","low","close",
    "volume","quote_volume","num_trades",
    "taker_buy_base","taker_buy_quote",
]
MIN_TS_MS = int(pd.Timestamp("2010-01-01").timestamp() * 1000)
MAX_TS_MS = int(pd.Timestamp("2030-01-01").timestamp() * 1000)

# -----------------------------
# Helpers
# -----------------------------

def _symbol_from_filename(path: Path) -> str:
    stem = path.stem  # e.g., BTCUSDT-1m-2020-01
    return stem.split("-")[0] if "-" in stem else stem


def _frame_from_parent(path: Path) -> str:
    return path.parent.name


def read_one_csv(path_str: str) -> Optional[pd.DataFrame]:
    path = Path(path_str)
    try:
        df = pd.read_csv(path, header=None, names=CSV_COLS, low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read {path.name}: {e}")
        return None

    if df is None or df.empty:
        print(f"⚠️ Empty CSV: {path.name} — skipping.")
        return None

    # Keep only required cols
    df = df[KEEP_COLS].copy()

    # Normalize epoch units (seconds → milliseconds if detected)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    vals = df["open_time"].dropna()
    if not vals.empty:
        if (vals < 1e12).mean() >= 0.8:
            unit_detected = "seconds"
            df["open_time"] *= 1000  # إلى ميلي ثانية
        elif (vals > 1e14).mean() >= 0.8:
            unit_detected = "microseconds"
            df["open_time"] = (df["open_time"] / 1000.0).round()  # إلى ميلي ثانية
        else:
            unit_detected = "milliseconds"
        df["open_time"] = df["open_time"].astype("Int64")
        print(f"ℹ️ {path.name}: detected {unit_detected}")

    # Bounds filter (milliseconds)
    df = df[(df["open_time"] >= MIN_TS_MS) & (df["open_time"] <= MAX_TS_MS)]
    if df.empty:
        # noisy files in 2025 seconds will be filtered here before conversion otherwise
        print(f"⚠️ {path.name}: no rows within time bounds — skipping.")
        return None

    # Standardize
    df.rename(columns={"open_time": "timestamp"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df[df["datetime"].notna()]
    if df.empty:
        print(f"⚠️ {path.name}: no valid datetimes — skipping.")
        return None

    df["symbol"] = _symbol_from_filename(path)
    df["frame"] = _frame_from_parent(path)

    for c in ["open","high","low","close","volume","quote_volume","num_trades","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32", copy=False)

    df["timestamp"] = df["timestamp"].astype("int64", copy=False)
    df["symbol"] = df["symbol"].astype("string")
    df["frame"] = df["frame"].astype("string")

    # Final order
    df = df[FINAL_COLS]
    return df


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_feather(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    try:
        df.to_feather(out_path, compression="lz4")
    except Exception:
        df.to_feather(out_path)

# -----------------------------
# Merge / Split
# -----------------------------

def _merge_frames_per_symbol(dfs: List[pd.DataFrame], frame: str, out_dir: Path) -> Dict[str, pd.DataFrame]:
    """Merge per-symbol from an already-parsed list of DataFrames.
    Returns a dict {symbol: merged_df} and also writes merged Feather files.
    """
    by_symbol: Dict[str, List[pd.DataFrame]] = {}
    for d in dfs:
        sym = d["symbol"].iat[0]
        by_symbol.setdefault(sym, []).append(d)

    merged_map: Dict[str, pd.DataFrame] = {}
    for sym, parts in by_symbol.items():
        merged = pd.concat(parts, ignore_index=True)
        if merged.empty:
            print(f"⚠️ [{frame}] {sym}: merged empty — skip.")
            continue
        merged.sort_values("datetime", inplace=True)
        merged.reset_index(drop=True, inplace=True)

        start = pd.to_datetime(merged["datetime"].iloc[0]).strftime("%Y-%m-%d")
        end   = pd.to_datetime(merged["datetime"].iloc[-1]).strftime("%Y-%m-%d")
        out_name = f"{sym}-{frame}-{start}_to_{end}.feather"
        out_path = out_dir / frame / out_name
        save_feather(merged[FINAL_COLS], out_path)
        merged_map[sym] = merged
        print(f"✅ [{frame}] {sym}: saved {out_name} rows={len(merged):,}")
    return merged_map


def _split_1s_by_year_from_merged(merged_map: Dict[str, pd.DataFrame], frame: str, out_dir: Path) -> None:
    for sym, merged in merged_map.items():
        if merged.empty:
            continue
        tmp = merged.copy()
        tmp["year"] = tmp["datetime"].dt.year
        years = tmp["year"].dropna().unique()
        if len(years) == 0:
            print(f"⚠️ [1s] {sym}: no year values — skip yearly split.")
            continue
        for y, part in tmp.groupby("year"):
            if part.empty:
                continue
            start = pd.to_datetime(part["datetime"].iloc[0]).strftime("%Y-%m-%d")
            end   = pd.to_datetime(part["datetime"].iloc[-1]).strftime("%Y-%m-%d")
            out_name = f"{sym}-{frame}-{start}_to_{end}.feather"
            out_path = out_dir / frame / out_name
            save_feather(part[FINAL_COLS], out_path)
        print(f"✅ [1s] {sym}: saved yearly parts = {tmp['year'].nunique()} year(s)")

# -----------------------------
# Frame pipeline
# -----------------------------

def process_frame_parallel(raw_dir: Path, out_dir: Path, frame: str, split_1s: str = "year") -> None:
    in_dir = raw_dir / frame
    if not in_dir.exists():
        print(f"⚠️ Skip {frame}: {in_dir} not found")
        return

    all_files = sorted(glob.glob(str(in_dir / "*.csv")))
    if not all_files:
        print(f"⚠️ No CSV files in {in_dir}, skipping {frame}.")
        return

    nprocs = min(60, cpu_count())  # Windows handle limit workaround
    chunk = max(1, len(all_files) // (nprocs * 4) or 1)
    print(f"[CPU] frame={frame} reading {len(all_files)} files with {nprocs} workers…")
    print(f"[CPU] frame={frame} using chunksize={chunk}")

    with Pool(processes=nprocs) as pool:
        dfs = list(pool.imap_unordered(read_one_csv, all_files, chunksize=chunk))

    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        print(f"⚠️ {frame}: no valid DataFrames after parsing — skip.")
        return

    # Step 1: Merge for ALL frames (including 1s)
    merged_map = _merge_frames_per_symbol(dfs, frame, out_dir)

    # Step 2: If frame is 1s and split mode is 'year', also split by year from merged
    if frame == "1s" and split_1s == "year":
        _split_1s_by_year_from_merged(merged_map, frame, out_dir)

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV → Feather (float32) compatible with TradingEnv, parallelized")
    p.add_argument("--raw-dir", default="data")
    p.add_argument("--out-dir", default="data_ready")
    p.add_argument("--frames", nargs="*", default=["1s","1m","3m","5m"])
    p.add_argument("--split-1s", choices=["year"], default="year")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    frames: List[str] = args.frames
    print(f"[CFG] raw_dir={raw_dir}  out_dir={out_dir}  frames={frames}")
    print(f"[CFG] 1s split: mode={args.split_1s}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        process_frame_parallel(raw_dir, out_dir, frame, split_1s=args.split_1s)
    print("🎯 Done →", out_dir.resolve())


if __name__ == "__main__":
    main()
