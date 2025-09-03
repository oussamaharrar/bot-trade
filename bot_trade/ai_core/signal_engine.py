from __future__ import annotations
import math
from typing import List, Tuple
import pandas as pd
from datetime import datetime, timezone

from .collectors.market_collector import collect_market
from .collectors.news_collector import collect_news
from .normalizers.scalers import z_score
from .enrichers.features import derive_features


def run_pipeline(df: pd.DataFrame, symbol: str, frame: str, emit_dummy: bool = False) -> Tuple[List[dict], set[str]]:
    """Run collectors → normalizers → enrichers and assemble signal records."""
    market = collect_market(df)
    news = collect_news(df)
    feats = derive_features(market, news)
    records: List[dict] = []
    sources: set[str] = set()
    ts_index = df.index
    for name, series in feats.items():
        series = z_score(series)
        if series is None:
            continue
        collector = 'news' if name == 'sentiment' else 'market'
        sources.add(collector)
        for ts, val in zip(ts_index, series):
            try:
                val_f = float(val)
            except Exception:
                continue
            if not math.isfinite(val_f):
                continue
            ts_iso = (
                ts if isinstance(ts, str)
                else pd.Timestamp(ts).to_pydatetime().astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
            )
            records.append({
                'ts': ts_iso,
                'symbol': symbol,
                'frame': frame,
                'source': 'ai_core',
                'signal': name,
                'value': val_f,
                'confidence': 0.5 if collector == 'news' else 1.0,
                'provenance': {
                    'collector': collector,
                    'features': [name],
                    'notes': None,
                },
            })
    if emit_dummy:
        ts_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        records.append({
            'ts': ts_iso,
            'symbol': symbol,
            'frame': frame,
            'source': 'ai_core',
            'signal': 'dummy_signal',
            'value': 0.0,
            'confidence': 1.0,
            'provenance': {
                'collector': 'market',
                'features': ['dummy_signal'],
                'notes': 'synthetic',
            },
        })
        sources.add('market')
    # drop invalid and dedupe
    seen = set()
    valid: List[dict] = []
    dropped = 0
    for rec in records:
        key = (rec['ts'], rec['symbol'], rec['signal'])
        if key in seen:
            continue
        seen.add(key)
        val_f = rec.get('value')
        if not math.isfinite(float(val_f)):
            dropped += 1
            continue
        valid.append(rec)
    if dropped:
        print(f"[AI_CORE] dropped_invalid count={dropped}")
    return valid, sources
