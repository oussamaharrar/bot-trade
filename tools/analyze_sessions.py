"""Session level analytics helpers.

The original script produced equity curve charts directly.  It now
exposes functions that return pandas DataFrames/Series so other tools
can reuse the logic.  The command line behaviour is kept for backward
compatibility."""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .analytics_common import (
    load_trades_df,
    compute_equity,
    compute_drawdown,
    compute_sharpe,
)


def analyse_session(base: str, symbol: str, frame: str) -> Dict[str, object]:
    """Load trades/steps and compute key metrics."""
    trades = load_trades_df(base, symbol, frame)
    if trades.empty:
        return {"trades": trades}
    equity = compute_equity(trades)
    drawdown = compute_drawdown(equity)
    sharpe = compute_sharpe(equity)
    return {
        "trades": trades,
        "equity": equity,
        "drawdown": drawdown,
        "sharpe": sharpe,
    }


def _save_curve(eq: pd.Series, frame: str, model: str) -> None:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    eq.to_csv(f"results/equity_curve_{frame}_{model}.csv", index=False)
    plt.figure()
    plt.plot(eq.values)
    plt.title(f"Equity Curve — {frame} — {model}")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(f"reports/equity_curve_{frame}_{model}.png", dpi=120)


def main():
    frame = os.getenv("CURRENT_FRAME", "unknown")
    model = os.getenv("CURRENT_MODEL", "deep_rl")
    symbol = os.getenv("CURRENT_SYMBOL", "BTCUSDT")
    base = os.getenv("RESULTS_BASE", "results")

    out = analyse_session(base, symbol, frame)
    trades, equity = out.get("trades"), out.get("equity")
    if trades is None or trades.empty:
        print("No results to analyze.")
        return
    _save_curve(equity, frame, model)
    dd = out.get("drawdown", pd.Series(dtype=float))
    sharpe = out.get("sharpe", 0.0)
    wins = (trades.get("pnl", pd.Series(0)).astype(float) > 0).sum()
    total = len(trades)
    winrate = wins / total if total else 0.0
    maxdd = float(dd.max()) if not dd.empty else 0.0
    with open("reports/leaderboard.md", "a", encoding="utf-8") as f:
        f.write(
            f"| {frame} | {model} | Sharpe {sharpe:.3f} | MaxDD {maxdd:.2%} | WinRate {winrate:.2%} |\n"
        )
    print(f"✅ جلسة التحليل مكتملة للفريم: {frame} — النموذج: {model}")


if __name__ == "__main__":
    main()
