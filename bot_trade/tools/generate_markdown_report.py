"""Minimal Markdown report generator for training sessions."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

import pandas as pd


def generate_summary(paths: Dict[str, str], symbol: str, frame: str) -> str:
    """Create a simple markdown summary under ``paths['reports']``.

    Parameters
    ----------
    paths: dict
        Mapping produced by ``config.rl_paths.build_paths``.
    symbol, frame: str
        Context for the report name.
    Returns
    -------
    str
        Path to the generated report file.
    """
    report_dir = paths.get("reports") or os.path.join(paths.get("results", ""), "reports")
    os.makedirs(report_dir, exist_ok=True)

    train_csv = paths.get("train_csv")
    eval_csv = paths.get("eval_csv")
    trades_csv = paths.get("trade_csv")

    lines = [f"# Training Summary {symbol}-{frame}", "", f"Generated: {datetime.utcnow().isoformat()}", ""]

    try:
        if train_csv and os.path.exists(train_csv):
            df = pd.read_csv(train_csv)
            lines.append(f"- train rows: {len(df)}")
    except Exception:
        pass
    try:
        if eval_csv and os.path.exists(eval_csv):
            df = pd.read_csv(eval_csv)
            if not df.empty and "value" in df.columns:
                lines.append(f"- best eval: {df['value'].max():.4f}")
    except Exception:
        pass
    try:
        if trades_csv and os.path.exists(trades_csv):
            df = pd.read_csv(trades_csv)
            lines.append(f"- trades: {len(df)}")
            if "pnl" in df.columns:
                lines.append(f"- cumulative pnl: {df['pnl'].sum():.4f}")
    except Exception:
        pass

    out_path = os.path.join(report_dir, f"{symbol}_{frame}_summary.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out_path


if __name__ == "__main__":  # pragma: no cover
    import argparse
    from bot_trade.config.rl_paths import get_paths

    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--frame", required=True)
    args = p.parse_args()
    paths = get_paths(args.symbol, args.frame)
    generate_summary(paths, args.symbol, args.frame)
