import pandas as pd
import ta

def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common trading indicators used by various strategies."""
    if 'close' not in df:
        if 'price' in df:
            df = df.rename(columns={'price': 'close'})
        else:
            raise ValueError("DataFrame must contain 'close' column")
    if 'volume' not in df:
        df['volume'] = 1.0

    df = df.copy()

    # RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_upper'] = bb.bollinger_hband()
    df['bollinger_lower'] = bb.bollinger_lband()
    df['bollinger_upper_diff'] = df['bollinger_upper'] - df['close']
    df['bollinger_lower_diff'] = df['close'] - df['bollinger_lower']

    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    return df
