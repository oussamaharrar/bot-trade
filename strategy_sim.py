import pandas as pd

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def apply_strategy(df):
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['RSI_14'] = calculate_rsi(df['close'])

    signals = []
    for i in range(14, len(df)):
        current_price = df['close'][i]
        prev_price = df['close'][i - 1]
        ema = df['EMA_5'][i]
        rsi = df['RSI_14'][i]
        price_change_pct = abs((current_price - prev_price) / prev_price) * 100

        # Ignore signals with low price movement or sideways RSI
        if price_change_pct < 0.2 or (45 <= rsi <= 55):
            signals.append((df['timestamp'][i], current_price, "HOLD"))
            continue

        if current_price > ema and rsi < 70 and current_price > prev_price:
            signals.append((df['timestamp'][i], current_price, "BUY"))
        elif current_price < ema and rsi > 30 and current_price < prev_price:
            signals.append((df['timestamp'][i], current_price, "SELL"))
        else:
            signals.append((df['timestamp'][i], current_price, "HOLD"))

    return signals
