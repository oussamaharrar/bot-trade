import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import ta

# -------- ML / Classic RL Data Preparation --------
RAW_DIR = "raw_binance"
OUTPUT_CSV = "training_dataset_full.csv"
OUTPUT_FEATHER = "training_dataset_full.feather"

def extract_symbol_from_filename(filename):
    base = os.path.basename(filename)
    match = base.split("-")[0]  # e.g., BTCUSDT-trades-2025-01.csv → BTCUSDT
    return match.upper() if match else "UNKNOWN"

def load_and_process_file(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = [
            'trade_id', 'price', 'qty', 'quote_qty', 'timestamp', 'is_buyer_maker', 'is_best_match'
        ]

        # Handle timestamp safely
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

        # Add technical indicators
        df_resampled['rsi'] = ta.momentum.RSIIndicator(df_resampled['close']).rsi()
        df_resampled['ema20'] = ta.trend.EMAIndicator(df_resampled['close'], window=20).ema_indicator()
        df_resampled['sma50'] = ta.trend.SMAIndicator(df_resampled['close'], window=50).sma_indicator()

        macd = ta.trend.MACD(df_resampled['close'])
        df_resampled['macd'] = macd.macd()
        df_resampled['macd_signal'] = macd.macd_signal()

        bb = ta.volatility.BollingerBands(df_resampled['close'])
        df_resampled['bb_bbm'] = bb.bollinger_mavg()
        df_resampled['bb_bbh'] = bb.bollinger_hband()
        df_resampled['bb_bbl'] = bb.bollinger_lband()

        df_resampled['atr'] = ta.volatility.AverageTrueRange(
            high=df_resampled['high'], low=df_resampled['low'], close=df_resampled['close']
        ).average_true_range()

        df_resampled['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df_resampled['close'], volume=df_resampled['volume']
        ).on_balance_volume()

        df_resampled['cci'] = ta.trend.CCIIndicator(
            high=df_resampled['high'], low=df_resampled['low'], close=df_resampled['close']
        ).cci()

        stoch = ta.momentum.StochasticOscillator(
            high=df_resampled['high'], low=df_resampled['low'], close=df_resampled['close']
        )
        df_resampled['stoch_k'] = stoch.stoch()
        df_resampled['stoch_d'] = stoch.stoch_signal()

        df_resampled['roc'] = ta.momentum.ROCIndicator(df_resampled['close']).roc()

        df_resampled.dropna(inplace=True)
        df_resampled['symbol'] = extract_symbol_from_filename(file_path)

        print(f"[OK] {file_path} → {len(df_resampled)} rows")
        return df_resampled

    except Exception as e:
        print(f"[SKIP] Error processing {file_path}: {e}")
        return pd.DataFrame()

def ml_main():
    all_data = []
    for file in tqdm(sorted(os.listdir(RAW_DIR)), desc="ML Data Prep"):
        path = os.path.join(RAW_DIR, file)
        if os.path.isfile(path) and path.endswith('.csv'):
            df = load_and_process_file(path)
            if not df.empty:
                all_data.append(df)

    if all_data:
        df_merged = pd.concat(all_data)
        df_merged.sort_index(inplace=True)
        df_merged.to_csv(OUTPUT_CSV)
        df_merged.reset_index().to_feather(OUTPUT_FEATHER)
        print(f"[DONE] ML CSV saved: {OUTPUT_CSV}")
        print(f"[DONE] ML Feather saved: {OUTPUT_FEATHER}")
    else:
        print("[WARN] No valid ML data processed.")

# -------- Deep RL Tick Feature Generation --------
INPUT_DIR = "raw_binance/main"
OUTPUT_DIR = "data/tick_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_tick_file(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        'trade_id', 'price', 'qty', 'quote_qty', 'timestamp',
        'is_buyer_maker', 'is_best_match'
    ]

    # Convert time
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)

    # Cast types
    df['price'] = df['price'].astype(float)
    df['qty'] = df['qty'].astype(float)
    df['quote_qty'] = df['quote_qty'].astype(float)

    # Derived signals
    df['price_change'] = df['price'].diff().fillna(0)
    df['price_velocity'] = df['price_change'].rolling(window=5).mean().fillna(0)
    df['rolling_mean_10'] = df['price'].rolling(window=10).mean().fillna(method='bfill')
    df['rolling_std_10'] = df['price'].rolling(window=10).std().fillna(method='bfill')
    df['z_score'] = (df['price'] - df['rolling_mean_10']) / df['rolling_std_10'].replace(0, np.nan)
    df['z_score'] = df['z_score'].fillna(0)

    # Buy/Sell pressure
    df['buyer_volume'] = df['qty'].where(df['is_buyer_maker'] == False, 0)
    df['seller_volume'] = df['qty'].where(df['is_buyer_maker'] == True, 0)
    df['buy_sell_ratio'] = (df['buyer_volume'].rolling(5).sum() + 1e-6) / (df['seller_volume'].rolling(5).sum() + 1e-6)

    df['volume'] = df['qty'].rolling(5).sum().fillna(0)

    # State indicators
    df['volatility'] = df['price'].rolling(window=20).std().fillna(0)
    df['momentum'] = df['price'].diff(periods=3).fillna(0)
    df['direction'] = np.where(df['price_change'] > 0, 1, np.where(df['price_change'] < 0, -1, 0))

    # Add symbol column
    symbol = os.path.basename(file_path).split("-")[0].upper()
    df['symbol'] = symbol

    # Final cleanup
    df = df.drop(columns=['trade_id', 'quote_qty', 'is_best_match'])
    df.dropna(inplace=True)

    return df

def deep_main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    for f in tqdm(files, desc="Deep RL Data Prep"):
        path = os.path.join(INPUT_DIR, f)
        try:
            df_processed = process_tick_file(path)
            out_name = f.replace(".csv", "_features.feather")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            float_cols = df_processed.select_dtypes(include='float64').columns
            df_processed[float_cols] = df_processed[float_cols].astype('float32', copy=False)
            df_processed.reset_index().to_feather(out_path)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

# -------- Argument Parser --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Binance datasets for ML or Deep RL training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ml', action='store_true', help="Run classic ML / RL data preparation")
    group.add_argument('--deep', action='store_true', help="Run Deep RL tick feature generation")
    args = parser.parse_args()

    if args.ml:
        ml_main()
    elif args.deep:
        deep_main()
