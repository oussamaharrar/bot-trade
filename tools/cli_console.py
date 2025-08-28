"""Interactive command-line console for querying training analytics."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

from .analytics_common import (
    load_reward_df,
    load_trades_df,
    load_steps_df,
    load_decisions_df,
    compute_equity,
    compute_drawdown,
    compute_sharpe,
    penalties_breakdown,
    signals_agg,
    plot_line,
    plot_area,
    plot_bar,
    table_to_image,
)

try:
    from prompt_toolkit import PromptSession
except Exception:
    PromptSession = None

import pandas as pd


def status(base, symbol, frame):
    trades = load_trades_df(base, symbol, frame)
    eq = compute_equity(trades)
    return {
        "steps": len(trades),
        "equity": float(eq.iloc[-1]) if not eq.empty else 0.0,
        "sharpe": compute_sharpe(eq),
    }


def handle_command(cmd, base, symbol, frame):
    parts = cmd.strip().split()
    if not parts:
        return False
    c = parts[0].lower()
    if c in {"quit", "exit"}:
        return True
    if c == "status":
        s = status(base, symbol, frame)
        print(s)
    elif c == "perf":
        trades = load_trades_df(base, symbol, frame)
        eq = compute_equity(trades)
        dd = compute_drawdown(eq)
        print({
            "sharpe": compute_sharpe(eq),
            "maxdd": float(dd.max()) if not dd.empty else 0.0,
        })
    elif c == "signals":
        decisions = load_decisions_df(base, symbol, frame)
        top = int(parts[2]) if len(parts) > 2 and parts[1] == 'top' else 20
        print(signals_agg(decisions, top))
    elif c == "export":
        # export charts out=<dir>
        if len(parts) >= 3 and parts[1] == 'charts':
            out = parts[2].split('=')[1]
            os.makedirs(out, exist_ok=True)
            reward = load_reward_df(base, symbol, frame)
            trades = load_trades_df(base, symbol, frame)
            steps = load_steps_df(base, symbol, frame)
            decisions = load_decisions_df(base, symbol, frame)
            eq = compute_equity(trades)
            dd = compute_drawdown(eq)
            pen = penalties_breakdown(steps)
            sig = signals_agg(decisions)
            plot_line(reward.iloc[:,0], os.path.join(out,'reward.png'), 'reward') if not reward.empty else None
            plot_line(eq, os.path.join(out,'equity.png'), 'equity')
            plot_line(dd, os.path.join(out,'drawdown.png'), 'drawdown')
            plot_area(pen, os.path.join(out,'penalties.png'), 'penalties')
            plot_bar(sig.set_index('signal'), os.path.join(out,'signals_top.png'), 'signals')
            table_to_image(pd.DataFrame({'kpi':['sharpe','maxdd'], 'value':[compute_sharpe(eq), dd.max() if not dd.empty else 0]}), os.path.join(out,'kpis_table.png'))
            print(f"exported to {out}")
    else:
        print("unknown command")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--frame', default='1m')
    ap.add_argument('--base', default='results')
    ap.add_argument('--history', action='store_true')
    args = ap.parse_args()

    if PromptSession and args.history:
        sess = PromptSession()
        while True:
            cmd = sess.prompt('> ')
            if handle_command(cmd, args.base, args.symbol, args.frame):
                break
    else:
        while True:
            try:
                cmd = input('> ')
            except EOFError:
                break
            if handle_command(cmd, args.base, args.symbol, args.frame):
                break


if __name__ == '__main__':
    main()
