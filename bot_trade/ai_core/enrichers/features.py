from __future__ import annotations
import pandas as pd


def derive_features(market: pd.DataFrame, news: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Derive simple features from collector outputs."""
    feats: dict[str, pd.Series] = {}
    if market is not None and 'returns' in market:
        feats['momentum'] = market['returns'].rolling(3).mean()
    if news is not None and 'sentiment' in news:
        feats['sentiment'] = news['sentiment']
    return feats
