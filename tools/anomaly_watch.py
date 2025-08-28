"""Lightweight watchdog to detect stalled training or metric anomalies."""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from tools.analytics_common import (
    load_reward_df,
    load_trades_df,
    compute_equity,
    wait_for_first_write,
    artifact_paths,
)

import numpy as np
import pandas as pd


def check_stall(paths, stalled_sec):
    now = time.time()
    for p in paths:
        if not os.path.exists(p):
            continue
        m = os.path.getmtime(p)
        if now - m > stalled_sec:
            return True
    return False


def check_nan_reward(reward_df):
    return reward_df.isna().any().any()


def check_drawdown(eq, threshold):
    if eq.empty:
        return False
    roll = eq.cummax()
    dd = (roll - eq) / roll.replace(0, np.nan)
    return dd.max() > threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--frame', default='1m')
    ap.add_argument('--base', default='results')
    ap.add_argument('--refresh', type=int, default=10)
    ap.add_argument('--stalled-sec', type=int, default=120)
    ap.add_argument('--dd-warn', type=float, default=0.15)
    ap.add_argument('--no-wait', action='store_true')
    args = ap.parse_args()

    log_file = os.path.join('exports', args.symbol, args.frame, 'alerts.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if not args.no_wait:
        wait_for_first_write(args.base, args.symbol, args.frame)

    paths = artifact_paths(args.base, args.symbol, args.frame)

    while True:
        reward = load_reward_df(args.base, args.symbol, args.frame)
        trades = load_trades_df(args.base, args.symbol, args.frame)
        eq = compute_equity(trades)
        alerts = []
        if check_stall(paths, args.stalled_sec):
            alerts.append('training stalled')
        if check_nan_reward(reward):
            alerts.append('NaN reward')
        if check_drawdown(eq, args.dd_warn):
            alerts.append('drawdown high')
        if alerts:
            msg = f"[{datetime.utcnow().isoformat()}] " + ", ".join(alerts)
            print(msg)
            with open(log_file,'a',encoding='utf-8') as fh:
                fh.write(msg+'\n')
        time.sleep(max(1,args.refresh))


if __name__ == '__main__':
    main()
