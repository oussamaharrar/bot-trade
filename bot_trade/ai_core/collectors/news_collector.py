from __future__ import annotations
import pandas as pd

def collect_news(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dummy news sentiment collector.

    Returns a DataFrame with a constant neutral sentiment column for demo purposes.
    """
    if df is None:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    out['sentiment'] = 0.0
    return out
