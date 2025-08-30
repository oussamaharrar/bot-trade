import pandas as pd
from pathlib import Path
from tqdm import tqdm

SOURCE = Path("extracted_csv")
DEST = Path("data")
FRAMES = {
    "1s": 5,
    "1m": 3
}

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

DTYPES = {
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
}


def process_file(path: Path):
    try:
        df = pd.read_csv(path, header=None, names=COLUMNS)
        df = df[[
            "open_time", "open", "high", "low", "close", "volume",
            "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote"
        ]]
        df = df.astype(DTYPES)
        df = df[(df["open_time"] >= 1280000000000) & (df["open_time"] <= 1920000000000)]
        df["timestamp"] = df["open_time"]
        df.drop(columns=["open_time"], inplace=True)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = path.stem.split("-")[0]
        df["frame"] = path.parent.name
        return df
    except Exception as e:
        print(f"[!] Failed on {path.name}: {e}")
        return None


def extract_frame(tf: str, limit: int):
    print(f"\n📄 Processing {tf}")
    in_dir = SOURCE / tf
    out_dir = DEST / tf
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csvs = sorted(list(in_dir.glob("*.csv")))
    dfs = []
    for path in tqdm(all_csvs, desc=f"Reading {tf}"):
        df = process_file(path)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print(f"[!] No valid data for {tf}")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.sort_values(by="datetime", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    chunk_size = len(full_df) // limit
    for i in range(limit):
        chunk = full_df.iloc[i * chunk_size:(i + 1) * chunk_size]
        if chunk.empty:
            continue
        symbol = chunk["symbol"].iloc[0]
        start = chunk["datetime"].min().strftime('%Y-%m-%d')
        end = chunk["datetime"].max().strftime('%Y-%m-%d')
        out_path = out_dir / f"{symbol}-{tf}-{start}_to_{end}.feather"
        chunk.to_feather(out_path)
        print(f"✅ {out_path.name} | rows: {len(chunk):,}")


if __name__ == "__main__":
    for tf, max_parts in FRAMES.items():
        extract_frame(tf, max_parts)
