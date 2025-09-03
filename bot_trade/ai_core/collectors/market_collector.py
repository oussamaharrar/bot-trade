from __future__ import annotations
import pandas as pd

def collect_market(df: pd.DataFrame) -> pd.DataFrame:
    """Return simple market-derived signals.

    Computes percentage returns if 'close' column exists.
    """
    if df is None or 'close' not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    out['returns'] = df['close'].pct_change().fillna(0.0)
    return out
