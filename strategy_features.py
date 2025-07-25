import pandas as pd
import ta

def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure necessary columns exist
    if 'close' not in df or 'volume' not in df:
        raise ValueError("DataFrame must contain 'close' and 'volume' columns")

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
