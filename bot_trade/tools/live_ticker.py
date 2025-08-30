"""Auto-refreshing CLI ticker showing training KPIs."""
from __future__ import annotations

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import argparse
import threading
from datetime import datetime, timezone

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
    wait_for_first_write,
)
from bot_trade.tools.paths import results_dir, logs_dir

try:
    from rich.console import Console
    from rich.live import Live
    RICH = True
except Exception:
    RICH = False
    Console = None

import pandas as pd


def build_display(base: str, symbol: str, frame: str, rollwin: int) -> str:
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


def maybe_images(base: str, symbol: str, frame: str, outdir: str, rollwin: int) -> None:
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
    if not reward.empty:
        plot_line(reward.iloc[:,0].rolling(rollwin).mean(), os.path.join(outdir,'reward.png'), 'reward')
    plot_line(eq, os.path.join(outdir,'equity.png'), 'equity')
    plot_line(dd, os.path.join(outdir,'drawdown.png'), 'drawdown')
    if not dd.empty:
        plot_line(pd.Series(dd.tail(rollwin).mean(), index=[0]), os.path.join(outdir,'sharpe.png'), 'sharpe')
    plot_area(pen, os.path.join(outdir,'penalties.png'), 'penalties')
    if not sig.empty:
        plot_bar(sig.set_index('signal'), os.path.join(outdir,'signals_top.png'), 'signals')
    table_to_image(pd.DataFrame({'kpi':['sharpe','maxdd'], 'value':[compute_sharpe(eq), dd.max() if not dd.empty else 0]}), os.path.join(outdir,'kpis_table.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--frame', default='1m')
    ap.add_argument('--refresh', type=int, default=10)
    ap.add_argument('--base')
    ap.add_argument('--run-id', required=True)  # NOTE: interface changed here - run_id now required to namespace outputs
    ap.add_argument('--images-out')
    ap.add_argument('--rollwin', type=int, default=200)
    ap.add_argument('--ansi', dest='ansi', action='store_true', default=True)
    ap.add_argument('--no-ansi', dest='ansi', action='store_false')
    ap.add_argument('--no-wait', action='store_true')
    args = ap.parse_args()

    base_dir = args.base or str(results_dir(args.symbol, args.frame))
    if not args.no_wait:
        wait_for_first_write(base_dir, args.symbol, args.frame)

    console = Console() if RICH and args.ansi else None
    stop_event = threading.Event()
    hb_path = logs_dir(args.symbol, args.frame, args.run_id) / "live_ticket_heartbeat.log"
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(hb_path, "a", encoding="utf-8") as hb:
            with Live(console=console, refresh_per_second=4) if console else nullcontext():
                while not stop_event.is_set():
                    try:
                        msg = build_display(base_dir, args.symbol, args.frame, args.rollwin)
                        if console:
                            console.clear()
                            console.print(msg)
                        else:
                            print(msg, flush=True)
                        if args.images_out:
                            maybe_images(base_dir, args.symbol, args.frame, args.images_out, args.rollwin)
                    except Exception as exc:  # noqa: BLE001 - want broad safety
                        err = f"live_ticker error: {exc}"
                        if console:
                            console.print(err)
                        else:
                            print(err, flush=True)
                    hb.write(datetime.now(timezone.utc).isoformat() + "\n")
                    hb.flush()
                    stop_event.wait(max(1, args.refresh))
    except KeyboardInterrupt:
        pass


# fallback context manager
from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield

if __name__ == '__main__':
    main()
