
import logging
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import ta
from joblib import Parallel, delayed

RAW_DIR = "raw_binance"
OUTPUT_DIR = "processed"
CHUNK_SIZE = 500_000  # عدد الصفوف لكل جزء

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)

def extract_symbol_from_filename(filename):
    base = os.path.basename(filename)
    return base.split("-")[0].upper() if "-" in base else "UNKNOWN"

def add_indicators(df):
    try:
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["sma50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_bbm"] = bb.bollinger_mavg()
        df["bb_bbh"] = bb.bollinger_hband()
        df["bb_bbl"] = bb.bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci()
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["roc"] = ta.momentum.ROCIndicator(df["close"]).roc()
    except Exception as e:
        logging.error(f"Indicator error: {e}")
    return df.dropna()

def process_file(file_path):
    logging.info(f"⏳ Loading {file_path}")
    df = pd.read_csv(file_path, header=None)
    df.columns = ['trade_id', 'price', 'qty', 'quote_qty', 'timestamp', 'is_buyer_maker', 'is_best_match']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df['price'] = df['price'].astype(float)
    df['qty'] = df['qty'].astype(float)

    df_resampled = df.resample('1Min').agg({
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum'
    })
    df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']
    df_resampled.dropna(inplace=True)

    symbol = extract_symbol_from_filename(file_path)
    chunks = [df_resampled.iloc[i:i+CHUNK_SIZE].copy() for i in range(0, len(df_resampled), CHUNK_SIZE)]

    def process_and_save(i, chunk):
        chunk = add_indicators(chunk)
        chunk["symbol"] = symbol
        output_name = f"{symbol}_part{i:02d}.feather"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        chunk.reset_index().to_feather(output_path)
        logging.info(f"✅ Saved {output_path} with {len(chunk)} rows")

    Parallel(n_jobs=8)(delayed(process_and_save)(i, chunk) for i, chunk in enumerate(chunks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to raw Binance CSV")
    args = parser.parse_args()

    process_file(args.file)
