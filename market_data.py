import ccxt
import pandas as pd

def fetch_ohlcv(symbol="C/USDT", timeframe='1m', limit=100):
    exchange = ccxt.binance()
    symbol = symbol.replace("/", "")  # Binance format
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df