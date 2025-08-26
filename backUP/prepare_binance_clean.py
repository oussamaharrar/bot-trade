import os
import pandas as pd
from tqdm import tqdm
import ta

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

def main():
    all_data = []
    for file in tqdm(sorted(os.listdir(RAW_DIR))):
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
        print(f"[DONE] Saved CSV: {OUTPUT_CSV}")
        print(f"[DONE] Saved Feather: {OUTPUT_FEATHER}")
    else:
        print("[WARN] No valid data processed.")

if __name__ == "__main__":
    main()
