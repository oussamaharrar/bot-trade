import argparse
import os
import json
from typing import Dict

import pandas as pd
import yaml

from .knowledge_hub import ingest_text, summarize, propose_config_edits


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
    p.add_argument("--results", required=True)
    p.add_argument("--agents", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--propose-config", action="store_true")
    args = p.parse_args()

    kb: Dict[str, Dict] = {}
    if os.path.exists(args.results):
        for sym in os.listdir(args.results):
            sym_dir = os.path.join(args.results, sym)
            if not os.path.isdir(sym_dir):
                continue
            for frame in os.listdir(sym_dir):
                frame_dir = os.path.join(sym_dir, frame)
                if not os.path.isdir(frame_dir):
                    continue
                key = f"{sym}:{frame}"
                summ = summarize_dir(frame_dir)
                kb[key] = summ
                ingest_text(json.dumps(summ), {"symbol": sym, "frame": frame, "type": "run_summary"})
                if args.summarize:
                    summarize(sym, frame, last_n=20)

    os.makedirs(os.path.dirname(args.to) or ".", exist_ok=True)
    with open(args.to, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)

    if args.propose_config:
        cfg_path = os.path.join('config', 'config.yaml')
        with open(cfg_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
        props = propose_config_edits(cfg)
        if props:
            with open(os.path.join('config', 'config.proposals.yaml'), 'w', encoding='utf-8') as fh:
                fh.write(yaml.safe_dump({'proposals': props}))


if __name__ == "__main__":
    main()
