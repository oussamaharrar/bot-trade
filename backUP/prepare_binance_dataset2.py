import pandas as pd
import argparse
import os
import math
from multiprocessing import Pool, cpu_count

# Tick CSV columns
TICK_COLUMNS = [
    'trade_id', 'price', 'qty', 'quote_qty',
    'timestamp', 'is_buyer_maker', 'is_best_match'
]

# Available timeframes
SUPPORTED_FRAMES = ['1s', '1min', '5min', '15min', '1h', '1d']
FREQ_MAP = {
    '1s': '1s',
    '1min': '1min',
    '5min': '5min',
    '15min': '15min',
    '1h': '1h',
    '1d': '1d'
}


def ohlcv_from_chunk(chunk: pd.DataFrame, freq: str) -> pd.DataFrame:
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='ms')
    chunk.set_index('timestamp', inplace=True)

    ohlc = chunk['price'].resample(freq).ohlc()
    volume = chunk['qty'].resample(freq).sum()
    df = ohlc.copy()
    df['volume'] = volume
    return df.dropna()


def process_file(input_path: str, output_base: str, chunk_size: int, timeframes: list):
    name, ext = os.path.splitext(os.path.basename(input_path))
    symbol = name.split('-')[0] if '-' in name else name
    print(f"\nProcessing file: {input_path}")

    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    total_chunks = math.ceil(total_lines / chunk_size)

    reader = pd.read_csv(
        input_path, header=None, names=TICK_COLUMNS,
        dtype={'trade_id': str, 'price': float, 'qty': float,
               'quote_qty': float, 'timestamp': int,
               'is_buyer_maker': bool, 'is_best_match': bool},
        chunksize=chunk_size
    )

    for idx, chunk in enumerate(reader, start=1):
        pct = (idx / total_chunks) * 100
        print(f"[{name}] chunk {idx}/{total_chunks} ({pct:.1f}%)")

        for tf in timeframes:
            df = ohlcv_from_chunk(chunk.copy(), FREQ_MAP[tf])
            if df.empty:
                continue
            ts = df.index[0].strftime('%Y-%m-%d')
            subdir = os.path.join(output_base, tf)
            os.makedirs(subdir, exist_ok=True)
            out_fname = f"{symbol}_{ts}_chunk{idx:04d}.feather"
            out_path = os.path.join(subdir, out_fname)
            df.to_feather(out_path)
            print(f"[{name}] Saved {tf} chunk {idx} -> {out_path}")


def process_directory(input_dir: str, output_dir: str, chunk_size: int, timeframes: list, symbol_filter: str = None):
    os.makedirs(output_dir, exist_ok=True)
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, f)) and
             (symbol_filter is None or f.startswith(symbol_filter))]
    print(f"Found {len(files)} files in {input_dir}, using {cpu_count()} cores...")

    args = [(f, output_dir, chunk_size, timeframes) for f in files]
    with Pool(cpu_count()) as pool:
        pool.starmap(process_file, args)


def main():
    parser = argparse.ArgumentParser(description='Convert tick CSVs into multi-frame OHLCV feather files.')
    parser.add_argument('--input_dir', default='raw_binance', help='Input directory of tick CSV files')
    parser.add_argument('--output_dir', default='data', help='Output directory for OHLCV files')
    parser.add_argument('--chunk_size', type=int, default=3_000_000, help='Rows per chunk')
    parser.add_argument('-f', '--frames', nargs='+', default=['1min'], choices=SUPPORTED_FRAMES,
                        help='List of timeframes to extract (e.g., 1s 1min 5min 1h 1d)')
    parser.add_argument('--symbol_filter', type=str, default=None,
                        help='Only process files that start with this symbol (e.g., BTCUSDT)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")

    process_directory(args.input_dir, args.output_dir, args.chunk_size, args.frames, args.symbol_filter)


if __name__ == '__main__':
    main()
