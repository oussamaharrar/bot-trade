"""Batch export of analytics charts and tables."""
from __future__ import annotations

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import argparse
import json
from datetime import datetime

from bot_trade.tools.analytics_common import (
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
    git_short_hash,
    wait_for_first_write,
)
from bot_trade.tools.paths import results_dir, report_dir

import pandas as pd

CHARTS = [
    "reward",
    "equity",
    "drawdown",
    "sharpe",
    "penalties",
    "trades",
    "signals",
    "tables",
    "exposure",
    "fees",
    "positions",
]


def export(base: str, symbol: str, frame: str, out_dir: str, charts, limit, rollwin, svg: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    reward = load_reward_df(base, symbol, frame, limit)
    trades = load_trades_df(base, symbol, frame, limit)
    steps = load_steps_df(base, symbol, frame, limit)
    decisions = load_decisions_df(base, symbol, frame, limit)
    eq = compute_equity(trades)
    dd = compute_drawdown(eq)
    pen = penalties_breakdown(steps)
    sig = signals_agg(decisions)
    meta = {"git": git_short_hash(), "generated": datetime.utcnow().isoformat()}

    def maybe_svg(path_png):
        if not svg:
            return
        base, _ = os.path.splitext(path_png)
        return base + '.svg'

    if "reward" in charts and not reward.empty:
        png = os.path.join(out_dir,'reward.png')
        plot_line(reward.iloc[:,0].rolling(rollwin).mean(), png, 'reward')
        if svg: plot_line(reward.iloc[:,0].rolling(rollwin).mean(), maybe_svg(png), 'reward')
    if "equity" in charts:
        png = os.path.join(out_dir,'equity.png')
        plot_line(eq, png, 'equity')
        if svg: plot_line(eq, maybe_svg(png), 'equity')
    if "drawdown" in charts:
        png = os.path.join(out_dir,'drawdown.png')
        plot_line(dd, png, 'drawdown')
        if svg: plot_line(dd, maybe_svg(png), 'drawdown')
    if "sharpe" in charts:
        s = compute_sharpe(eq)
        png = os.path.join(out_dir,'sharpe.png')
        table_to_image(pd.DataFrame({'sharpe':[s]}), png)
        if svg: table_to_image(pd.DataFrame({'sharpe':[s]}), maybe_svg(png))
    if "penalties" in charts:
        png = os.path.join(out_dir,'penalties.png')
        plot_area(pen, png, 'penalties')
        if svg: plot_area(pen, maybe_svg(png), 'penalties')
    if "signals" in charts:
        png = os.path.join(out_dir,'signals_top.png')
        plot_bar(sig.set_index('signal'), png, 'signals')
        if svg: plot_bar(sig.set_index('signal'), maybe_svg(png), 'signals')
    if "tables" in charts:
        png = os.path.join(out_dir,'kpis_table.png')
        table_to_image(pd.DataFrame({'kpi':['sharpe','maxdd'], 'value':[compute_sharpe(eq), dd.max() if not dd.empty else 0]}), png)
        if svg: table_to_image(pd.DataFrame({'kpi':['sharpe','maxdd'], 'value':[compute_sharpe(eq), dd.max() if not dd.empty else 0]}), maybe_svg(png))
    if "exposure" in charts and not trades.empty and 'qty' in trades.columns:
        png = os.path.join(out_dir,'exposure.png')
        plot_line(trades['qty'].abs().cumsum(), png, 'exposure')
        if svg: plot_line(trades['qty'].abs().cumsum(), maybe_svg(png), 'exposure')
    if "fees" in charts and not trades.empty and 'fee' in trades.columns:
        png = os.path.join(out_dir,'fees.png')
        plot_line(trades['fee'].cumsum(), png, 'fees')
        if svg: plot_line(trades['fee'].cumsum(), maybe_svg(png), 'fees')
    if "positions" in charts and not trades.empty and 'qty' in trades.columns:
        png = os.path.join(out_dir,'positions_timeline.png')
        plot_line(trades['qty'].cumsum(), png, 'positions')
        if svg: plot_line(trades['qty'].cumsum(), maybe_svg(png), 'positions')
    with open(os.path.join(out_dir,'index.json'),'w',encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--frame', default='1m')
    ap.add_argument('--base')
    ap.add_argument('--run-id', required=True)  # NOTE: interface changed here - run_id is required
    ap.add_argument('--out')
    ap.add_argument('--charts', default=','.join(CHARTS))
    ap.add_argument('--svg', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--rollwin', type=int, default=200)
    ap.add_argument('--no-wait', action='store_true')
    args = ap.parse_args()

    base_dir = args.base or str(results_dir(args.symbol, args.frame))
    out_dir = args.out or str(report_dir(args.symbol, args.frame, args.run_id))

    if not args.no_wait:
        wait_for_first_write(base_dir, args.symbol, args.frame)

    charts = [c.strip() for c in args.charts.split(',') if c.strip()]
    export(base_dir, args.symbol, args.frame, out_dir, charts, args.limit, args.rollwin, svg=args.svg)


if __name__ == '__main__':
    main()
