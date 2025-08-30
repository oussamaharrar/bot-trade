"""Lightweight watchdog to detect stalled training or metric anomalies."""
from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime, timezone

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import json

from bot_trade.tools.analytics_common import (
    load_reward_df,
    load_trades_df,
    compute_equity,
    wait_for_first_write,
    artifact_paths,
)
from bot_trade.tools.paths import results_dir, report_dir, logs_dir
from bot_trade.tools.runctx import atomic_write_json

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
    ap.add_argument('--base')
    ap.add_argument('--refresh', type=int, default=10)
    ap.add_argument('--stalled-sec', type=int, default=120)
    ap.add_argument('--dd-warn', type=float, default=0.15)
    ap.add_argument('--no-wait', action='store_true')
    ap.add_argument('--run-id', required=True)  # NOTE: interface changed here - run_id now required to namespace outputs
    args = ap.parse_args()

    base_dir = args.base or str(results_dir(args.symbol, args.frame))
    if not args.no_wait:
        wait_for_first_write(base_dir, args.symbol, args.frame)

    paths = artifact_paths(base_dir, args.symbol, args.frame)
    log_dir = logs_dir(args.symbol, args.frame, args.run_id)
    rep_dir = report_dir(args.symbol, args.frame, args.run_id)
    log_file = log_dir / "anomalies.log"
    rep_file = rep_dir / "anomalies.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    try:
        while not stop_event.is_set():
            reward = load_reward_df(base_dir, args.symbol, args.frame)
            trades = load_trades_df(base_dir, args.symbol, args.frame)
            eq = compute_equity(trades)
            anomalies = []
            if check_stall(paths, args.stalled_sec):
                anomalies.append({
                    "name": "training_stalled",
                    "value": args.stalled_sec,
                    "severity": "critical",
                    "source": "artifact_mtime",
                })
            if check_nan_reward(reward):
                anomalies.append({
                    "name": "nan_reward",
                    "value": 1,
                    "severity": "high",
                    "source": "reward_csv",
                })
            if check_drawdown(eq, args.dd_warn):
                anomalies.append({
                    "name": "drawdown_high",
                    "value": float(args.dd_warn),
                    "severity": "warn",
                    "source": "equity",
                })
            ts = datetime.now(timezone.utc).isoformat()
            if anomalies:
                print(f"[{ts}] anomalies detected: {anomalies}")
            else:
                print(f"[{ts}] none detected")
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": ts, "anomalies": anomalies}) + "\n")
            atomic_write_json(rep_file, {"ts": ts, "anomalies": anomalies})
            stop_event.wait(max(1, args.refresh))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
