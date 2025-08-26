import os
import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_DIR = "raw_binance/main"
OUTPUT_DIR = "data/tick_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_tick_file(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        'trade_id', 'price', 'qty', 'quote_qty', 'timestamp',
        'is_buyer_maker', 'is_best_match'
    ]

    # تحويل الوقت
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)

    # تحويل الأنواع
    df['price'] = df['price'].astype(float)
    df['qty'] = df['qty'].astype(float)
    df['quote_qty'] = df['quote_qty'].astype(float)

    # إشارات مشتقة
    df['price_change'] = df['price'].diff().fillna(0)
    df['price_velocity'] = df['price_change'].rolling(window=5).mean().fillna(0)
    df['rolling_mean_10'] = df['price'].rolling(window=10).mean().fillna(method='bfill')
    df['rolling_std_10'] = df['price'].rolling(window=10).std().fillna(method='bfill')
    df['z_score'] = (df['price'] - df['rolling_mean_10']) / df['rolling_std_10'].replace(0, np.nan)
    df['z_score'] = df['z_score'].fillna(0)

    # ضغط البيع والشراء
    df['buyer_volume'] = df['qty'].where(df['is_buyer_maker'] == False, 0)
    df['seller_volume'] = df['qty'].where(df['is_buyer_maker'] == True, 0)
    df['buy_sell_ratio'] = (df['buyer_volume'].rolling(5).sum() + 1e-6) / (df['seller_volume'].rolling(5).sum() + 1e-6)

    df['volume'] = df['qty'].rolling(5).sum().fillna(0)

    # مؤشرات لتحديد "حالات"
    df['volatility'] = df['price'].rolling(window=20).std().fillna(0)
    df['momentum'] = df['price'].diff(periods=3).fillna(0)
    df['direction'] = np.where(df['price_change'] > 0, 1, np.where(df['price_change'] < 0, -1, 0))

    # تنظيف النهائي
    df = df.drop(columns=['trade_id', 'quote_qty', 'is_best_match'])
    df.dropna(inplace=True)

    return df

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    for f in tqdm(files, desc="Processing tick files"):
        path = os.path.join(INPUT_DIR, f)
        try:
            df_processed = process_tick_file(path)
            out_name = f.replace(".csv", "_features.feather")
            out_path = os.path.join(OUTPUT_DIR, out_name)
            df_processed.reset_index().to_feather(out_path)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

if __name__ == "__main__":
    main()
