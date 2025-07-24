import ccxt
import pandas as pd
import os
from strategy_sim import apply_strategy
from results_logger import simulate_wallet

def fetch_historical(symbol, timeframe, limit=1000):
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol.replace("/", ""), timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def run_historical_sim(symbol, timeframe):
    df = fetch_historical(symbol, timeframe)
    actions = apply_strategy(df)
    logs, final_value = simulate_wallet(actions, initial_usdt=1000.0)
    print(f"[{symbol} - {timeframe}] ✅ Simulation complete. Final: {final_value:.2f} USDT")
    return logs, final_value

if __name__ == "__main__":
    symbols = ["SOL/USDT", "ETH/USDT", "BTC/USDT", "C/USDT", "BNB/USDT"]
    timeframes = ["1m", "5m", "15m"]
    
    for symbol in symbols:
        for tf in timeframes:
            try:
                run_historical_sim(symbol, tf)
            except Exception as e:
                print(f"[{symbol} - {tf}] ❌ Error: {e}")