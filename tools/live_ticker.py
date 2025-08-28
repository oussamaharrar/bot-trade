"""Auto-refreshing CLI ticker showing training KPIs."""
from __future__ import annotations

import argparse
import os
import time
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
    from rich.console import Console
    from rich.live import Live
    RICH = True
except Exception:
    RICH = False
    Console = None

import pandas as pd


def build_display(base, symbol, frame, rollwin):
    reward = load_reward_df(base, symbol, frame)
    steps = load_steps_df(base, symbol, frame)
    trades = load_trades_df(base, symbol, frame)
    eq = compute_equity(trades)
    dd = compute_drawdown(eq)
    sharpe = compute_sharpe(eq)
    out_lines = []
    if not steps.empty:
        step = int(steps.index.max()) + 1
        out_lines.append(f"steps: {step}")
    if not reward.empty:
        col = reward.columns.intersection(['ep_rew_mean','reward']).tolist()[0]
        out_lines.append(f"reward_last: {reward[col].tail(rollwin).mean():.3f}")
    if not eq.empty:
        out_lines.append(f"equity: {eq.iloc[-1]:.2f} sharpe: {sharpe:.2f}")
    if not dd.empty:
        out_lines.append(f"maxdd: {dd.max():.2%}")
    return "\n".join(out_lines)


def maybe_images(base, symbol, frame, outdir, rollwin):
    if not outdir:
        return
    reward = load_reward_df(base, symbol, frame)
    trades = load_trades_df(base, symbol, frame)
    steps = load_steps_df(base, symbol, frame)
    decisions = load_decisions_df(base, symbol, frame)
    eq = compute_equity(trades)
    dd = compute_drawdown(eq)
    pen = penalties_breakdown(steps)
    sig = signals_agg(decisions)
    plot_line(reward.iloc[:,0].rolling(rollwin).mean(), os.path.join(outdir,'reward.png'), 'reward') if not reward.empty else None
    plot_line(eq, os.path.join(outdir,'equity.png'), 'equity')
    plot_line(dd, os.path.join(outdir,'drawdown.png'), 'drawdown')
    plot_line(pd.Series(dd.tail(rollwin).mean(), index=[0]), os.path.join(outdir,'sharpe.png'), 'sharpe')
    plot_area(pen, os.path.join(outdir,'penalties.png'), 'penalties')
    plot_bar(sig.set_index('signal'), os.path.join(outdir,'signals_top.png'), 'signals')
    table_to_image(pd.DataFrame({'kpi':['sharpe','maxdd'], 'value':[compute_sharpe(eq), dd.max() if not dd.empty else 0]}), os.path.join(outdir,'kpis_table.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--frame', default='1m')
    ap.add_argument('--refresh', type=int, default=10)
    ap.add_argument('--base', default='results')
    ap.add_argument('--images-out')
    ap.add_argument('--rollwin', type=int, default=200)
    ap.add_argument('--ansi', dest='ansi', action='store_true', default=True)
    ap.add_argument('--no-ansi', dest='ansi', action='store_false')
    args = ap.parse_args()

    console = Console() if RICH and args.ansi else None
    last = ''
    with Live(console=console, refresh_per_second=4) if console else nullcontext():
        while True:
            msg = build_display(args.base, args.symbol, args.frame, args.rollwin)
            if console:
                console.clear()
                console.print(msg)
            else:
                print(msg, flush=True)
            if args.images_out:
                maybe_images(args.base, args.symbol, args.frame, args.images_out, args.rollwin)
            time.sleep(max(1, args.refresh))


# fallback context manager
from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield

if __name__ == '__main__':
    main()
