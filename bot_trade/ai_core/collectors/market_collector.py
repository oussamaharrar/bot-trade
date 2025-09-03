from __future__ import annotations
import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average helper."""
    return series.ewm(span=span, adjust=False).mean()


def collect_market(df: pd.DataFrame) -> pd.DataFrame:
    """Return MACD, RSI and realized volatility features."""
    if df is None or "close" not in df.columns:
        return pd.DataFrame()
    price = pd.to_numeric(df["close"], errors="coerce")
    ema12 = _ema(price, 12)
    ema26 = _ema(price, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll = 14
    avg_gain = up.ewm(alpha=1 / roll, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / roll, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    log_ret = np.log(price.replace(0, np.nan)).diff()
    rv = log_ret.rolling(window=min(30, len(log_ret)), min_periods=1).std().clip(upper=1.0)
    out = pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "rsi": rsi,
            "rv": rv,
        },
        index=df.index,
    )
    return out
