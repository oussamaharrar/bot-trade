
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import pyarrow.feather as feather
import os

FRAME_SPLITS = {
    "30m": 1,
    "10m": 2,
    "5m": 2,
    "3m": 2,
    "1m": 3,
    "1s": 9
}

DATA_DIR = Path("data")
LOGS_DIR = Path("merge_logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def extract_dates(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return datetime.strptime(match[1], "%Y-%m-%d"), datetime.strptime(match[2], "%Y-%m-%d")
    return None, None

def split_ranges(start, end, n_parts):
    total_days = (end - start).days
    step = total_days // n_parts
    return [(start + pd.Timedelta(days=i*step), start + pd.Timedelta(days=(i+1)*step)) for i in range(n_parts - 1)] + [(start + pd.Timedelta(days=(n_parts-1)*step), end)]

def process_frame(frame, n_parts):
    log_lines = [f"# Merge log for frame: {frame}"]
    frame_dir = DATA_DIR / frame
    if not frame_dir.exists():
        log_lines.append(f"[SKIP] Frame folder {frame_dir} not found.")
        return log_lines

    files = sorted(list(frame_dir.glob("*.feather")))
    if not files:
        log_lines.append(f"[SKIP] No files in {frame_dir}")
        return log_lines

    files_by_symbol = {}
    for file in files:
        symbol = file.name.split("-")[0]
        files_by_symbol.setdefault(symbol, []).append(file)

    for symbol, symbol_files in files_by_symbol.items():
        date_ranges = [extract_dates(f.name) for f in symbol_files]
        valid = [(f, s, e) for f, (s, e) in zip(symbol_files, date_ranges) if s and e]
        if not valid:
            log_lines.append(f"[WARN] No valid date-ranged files for {symbol}")
            continue

        start = min(v[1] for v in valid)
        end = max(v[2] for v in valid)
        log_lines.append(f"[INFO] {symbol} range: {start.date()} to {end.date()}")

        splits = split_ranges(start, end, n_parts)

        for i, (s, e) in enumerate(splits):
            group = [f for f, fs, fe in valid if not (fe < s or fs > e)]
            if not group:
                log_lines.append(f"[WARN] No files in part {i+1} ({s.date()} to {e.date()}) for {symbol}")
                continue

            dfs = [feather.read_feather(f) for f in group]
            combined = pd.concat(dfs).sort_values(by=dfs[0].columns[0])
            out_name = f"{symbol}-{frame}-{s.date()}_to_{e.date()}.feather"
            out_path = frame_dir / out_name
            combined.reset_index(drop=True).to_feather(out_path)
            log_lines.append(f"[MERGED] {len(group)} files → {out_name}")

    return log_lines

all_logs = []
for frame, splits in FRAME_SPLITS.items():
    logs = process_frame(frame, splits)
    all_logs.extend(logs)

with open(LOGS_DIR / "merge_summary.log", "w", encoding="utf-8") as f:
    f.write("\n".join(all_logs))

print("✅ Merging complete. See log: merge_logs/merge_summary.log")
