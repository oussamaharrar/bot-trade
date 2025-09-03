"""Generate a synthetic OHLCV dataset for development and tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def generate(symbol: str, frame: str, out_dir: Path) -> Path:
    rng = np.random.default_rng(0)
    periods = 90 * 24 * 60  # 90 days of minutes
    start = pd.Timestamp.utcnow() - pd.Timedelta(minutes=periods)
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    price = 100 + np.cumsum(rng.normal(0, 0.5, periods))
    df = pd.DataFrame({
        "datetime": idx,
        "open": price,
        "high": price + rng.random(periods),
        "low": price - rng.random(periods),
        "close": price + rng.normal(0, 0.2, periods),
        "volume": rng.random(periods) * 10,
        "spread": np.abs(rng.normal(0.05, 0.01, periods)),
        "volatility": np.abs(rng.normal(0.5, 0.1, periods)),
        "depth": rng.random(periods) * 5,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = out_dir / frame
    sub.mkdir(parents=True, exist_ok=True)
    dest = sub / f"{symbol}-{frame}-synth.feather"
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_feather(tmp)
    os.replace(tmp, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate synthetic OHLCV data for development",
        epilog="Example: python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready",
    )
    ap.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    ap.add_argument("--frame", default="1m", help="Time frame")
    ap.add_argument("--out", default="data_ready", help="Output data directory")
    ns = ap.parse_args(argv)

    dest = generate(ns.symbol, ns.frame, Path(ns.out))
    print(dest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from bot_trade.config.encoding import force_utf8

    force_utf8()
    raise SystemExit(main())

