import os
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ✅ تحديد المجلد الأساسي (جذر المشروع)
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ تحديث المسارات بناءً على الجذر
RAW_DIR = BASE_DIR / "raw_binance"
EXTRACTED_DIR = BASE_DIR / "extracted_csv"
FINAL_DIR = BASE_DIR / "data"
LOG_FILE = BASE_DIR / "data_process_log.txt"

MERGE_FRAMES = [
    "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo",
    "3m", "5m", "15m", "30m"
]
SKIP_MERGE_FRAMES = ["1s", "1m"]

os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

# ---------- فك الضغط ---------- #
def unzip_all():
    for tf in MERGE_FRAMES + SKIP_MERGE_FRAMES:
        folder = RAW_DIR / tf
        out_folder = EXTRACTED_DIR / tf
        out_folder.mkdir(parents=True, exist_ok=True)
        zip_files = list(folder.glob("*.zip"))
        for zf in tqdm(zip_files, desc=f"Extracting {tf}"):
            try:
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall(out_folder)
            except Exception as e:
                print(f"[!] Failed to unzip {zf.name}: {e}")

# ---------- معالجة ملف واحد ---------- #
def process_csv(path: Path):
    try:
        df = pd.read_csv(path, header=None, names=CSV_COLUMNS)
        df = df[[
            "open_time", "open", "high", "low", "close", "volume",
            "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote"
        ]]
        df = df.astype({
            "open_time": "int64",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
            "quote_volume": "float32",
            "num_trades": "int32",
            "taker_buy_base": "float32",
            "taker_buy_quote": "float32"
        })

        min_ts = pd.Timestamp("2010-01-01").timestamp() * 1000
        max_ts = pd.Timestamp("2030-01-01").timestamp() * 1000
        df = df[(df["open_time"] >= min_ts) & (df["open_time"] <= max_ts)]

        df["timestamp"] = df["open_time"]
        df.drop(columns=["open_time"], inplace=True)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = path.stem.split("-")[0]
        df["frame"] = path.parent.name
        return df
    except Exception as e:
        print(f"[!] Error in {path.name}: {e}")
        return None

# ---------- دمج مجلد زمني ---------- #
def merge_timeframe(tf):
    print(f"\n📦 Merging {tf}")
    in_folder = EXTRACTED_DIR / tf
    out_folder = FINAL_DIR / tf
    out_folder.mkdir(parents=True, exist_ok=True)

    all_csvs = list(in_folder.glob("*.csv"))
    with Pool(cpu_count()) as pool:
        dfs = list(tqdm(pool.imap(process_csv, all_csvs), total=len(all_csvs)))
    dfs = [df for df in dfs if df is not None]
    if not dfs:
        print(f"[!] No valid data for {tf}")
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged.sort_values(by=["symbol", "datetime"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    start = merged['datetime'].min().strftime('%Y-%m-%d')
    end = merged['datetime'].max().strftime('%Y-%m-%d')
    example_symbol = merged['symbol'].iloc[0]

    out_path = out_folder / f"{example_symbol}-{tf}-{start}_to_{end}.feather"
    merged.to_feather(out_path)
    print(f"✅ Saved {out_path.name}: {len(merged):,} rows")

    with open(LOG_FILE, "a") as log:
        log.write(f"{out_path} | rows: {len(merged)}\n")

# ---------- حفظ كل ملف بدون دمج للفريمات الصغيرة ---------- #
def export_limited(tf, max_files):
    print(f"\n📄 Exporting max {max_files} files for {tf}")
    in_folder = EXTRACTED_DIR / tf
    out_folder = FINAL_DIR / tf
    out_folder.mkdir(parents=True, exist_ok=True)

    all_csvs = sorted(list(in_folder.glob("*.csv")))[:max_files]
    for path in tqdm(all_csvs):
        df = process_csv(path)
        if df is None or df.empty:
            continue

        symbol = df['symbol'].iloc[0]
        start = df['datetime'].min().strftime('%Y-%m-%d')
        end = df['datetime'].max().strftime('%Y-%m-%d')
        out_name = f"{symbol}-{tf}-{start}_to_{end}.feather"
        out_path = out_folder / out_name
        df.to_feather(out_path)

        with open(LOG_FILE, "a") as log:
            log.write(f"{out_path} | rows: {len(df)}\n")

# ---------- تشغيل كامل ---------- #
def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    unzip_all()
    for tf in MERGE_FRAMES:
        merge_timeframe(tf)
    export_limited("1s", 5)
    export_limited("1m", 3)

if __name__ == "__main__":
    main()
