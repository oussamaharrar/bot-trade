from __future__ import annotations
import pandas as pd


def derive_features(market: pd.DataFrame, news: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Pass through market features; news ignored for dummy signals."""
    feats: dict[str, pd.Series] = {}
    if market is not None:
        for col in market.columns:
            feats[col] = market[col]
    return feats
