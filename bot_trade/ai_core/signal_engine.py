from __future__ import annotations
import math
from typing import List, Tuple
import pandas as pd
from datetime import timezone

from .collectors.market_collector import collect_market
from .normalizers.scalers import z_score
from .enrichers.features import derive_features


def run_pipeline(
    df: pd.DataFrame, symbol: str, frame: str, emit_dummy: bool = False
) -> Tuple[List[dict], set[str]]:
    """Run collectors → normalizers → enrichers and assemble signal records."""
    if df is None or not emit_dummy:
        return [], set()
    market = collect_market(df)
    feats = derive_features(market)
    records: List[dict] = []
    sources: set[str] = set()
    ts_index = df.get("datetime", df.index)
    limit = min(len(ts_index), 2048)
    ts_index = ts_index[-limit:]
    dropped = 0
    seen = set()
    for name, series in feats.items():
        series = z_score(series)
        if series is None:
            continue
        series = series.tail(limit)
        sources.add("market")
        for ts, val in zip(ts_index, series):
            try:
                val_f = float(val)
            except Exception:
                dropped += 1
                continue
            if not math.isfinite(val_f):
                dropped += 1
                continue
            key = (ts, symbol, name)
            if key in seen:
                continue
            seen.add(key)
            ts_iso = (
                ts
                if isinstance(ts, str)
                else pd.Timestamp(ts)
                .to_pydatetime()
                .astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            records.append(
                {
                    "ts": ts_iso,
                    "symbol": symbol,
                    "frame": frame,
                    "source": "ai_core",
                    "signal": name,
                    "value": val_f,
                    "confidence": 1.0,
                    "provenance": {
                        "collector": "market",
                        "features": ["MACD", "RSI", "RV"],
                    },
                }
            )
    if dropped:
        print(f"[AI_CORE] dropped_invalid count={dropped}")
    return records, sources
