import argparse
import os
import json
from typing import Dict

import pandas as pd


def summarize_dir(dir_path: str) -> Dict:
    summary: Dict[str, any] = {}
    train_csv = os.path.join(dir_path, "train_log.csv")
    eval_csv = os.path.join(dir_path, "evaluation.csv")
    trades_csv = os.path.join(dir_path, "deep_rl_trades.csv")
    if os.path.exists(train_csv):
        try:
            df = pd.read_csv(train_csv)
            summary["train_rows"] = int(len(df))
        except Exception:
            pass
    if os.path.exists(eval_csv):
        try:
            df = pd.read_csv(eval_csv)
            if not df.empty and "value" in df.columns:
                summary["best_eval"] = float(df["value"].max())
        except Exception:
            pass
    if os.path.exists(trades_csv):
        try:
            df = pd.read_csv(trades_csv)
            summary["num_trades"] = int(len(df))
            if "pnl" in df.columns and not df.empty:
                summary["avg_trade_pnl"] = float(df["pnl"].mean())
        except Exception:
            pass
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--agents-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    kb: Dict[str, Dict] = {}
    for sym in os.listdir(args.results_dir):
        sym_dir = os.path.join(args.results_dir, sym)
        if not os.path.isdir(sym_dir):
            continue
        for frame in os.listdir(sym_dir):
            frame_dir = os.path.join(sym_dir, frame)
            if not os.path.isdir(frame_dir):
                continue
            key = f"{sym}:{frame}"
            kb[key] = summarize_dir(frame_dir)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)


if __name__ == "__main__":
    main()
