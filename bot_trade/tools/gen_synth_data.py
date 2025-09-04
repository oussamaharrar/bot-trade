"""Generate a synthetic OHLCV dataset for development and tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bot_trade.data.store_parquet import write_parquet_atomic
from bot_trade.data.validators import detect_gaps, detect_duplicates


def generate(symbol: str, frame: str, out_dir: Path, days: int = 90) -> Path:
    """Generate synthetic OHLCV data and store as Parquet."""

    rng = np.random.default_rng(0)
    periods = days * 24 * 60  # minutes
    start = pd.Timestamp.utcnow() - pd.Timedelta(minutes=periods)
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    price = 100 + np.cumsum(rng.normal(0, 0.5, periods))
    df = pd.DataFrame({
        "ts": idx.view("int64"),
        "open": price.astype("float32"),
        "high": (price + rng.random(periods)).astype("float32"),
        "low": (price - rng.random(periods)).astype("float32"),
        "close": (price + rng.normal(0, 0.2, periods)).astype("float32"),
        "volume": (rng.random(periods) * 10).astype("float32"),
        "symbol": symbol,
        "frame": frame,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = out_dir / frame
    sub.mkdir(parents=True, exist_ok=True)
    dest = sub / f"{symbol}-{frame}-synth.parquet"
    write_parquet_atomic(dest, df)
    gaps = detect_gaps(df, frame)
    dups = detect_duplicates(df)
    print(
        f"[DATA] out_dir={sub} file={dest} rows={len(df)} gaps={gaps} dups={dups}"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate synthetic OHLCV data for development",
        epilog="Example: python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready",
    )
    ap.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    ap.add_argument("--frame", default="1m", help="Time frame")
    ap.add_argument("--out", default="data_ready", help="Output data directory")
    ap.add_argument("--days", type=int, default=90, help="Number of days to simulate")
    ns = ap.parse_args(argv)

    dest = generate(ns.symbol, ns.frame, Path(ns.out), days=int(ns.days))
    print(dest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from bot_trade.tools.force_utf8 import force_utf8

    force_utf8()
    raise SystemExit(main())

