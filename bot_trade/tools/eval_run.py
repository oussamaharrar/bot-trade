#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic evaluation for RL runs.

Generates basic performance metrics and artifacts from existing logs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

from bot_trade.config.rl_paths import RunPaths, DEFAULT_REPORTS_DIR
from bot_trade.tools.atomic_io import write_json, write_png, write_text
from bot_trade.tools.latest import latest_run
from bot_trade.tools._headless import ensure_headless_once
from bot_trade.tools import export_charts




def evaluate_run(symbol: str, frame: str, run_id: str) -> Dict[str, Any]:
    """Compute evaluation metrics and equity/drawdown series."""

    import pandas as pd
    import matplotlib.pyplot as plt
    from bot_trade.eval import metrics
    from bot_trade.eval.utils import load_reward_log, load_trades
    from bot_trade.eval.equity import build_equity_drawdown

    rp = RunPaths(symbol, frame, str(run_id))
    perf_dir = rp.performance_dir

    reward_df = load_reward_log(rp.logs)
    trades = load_trades(rp.logs)
    equity, drawdown = build_equity_drawdown(rp.logs)
    metrics_dict = metrics.compute_all(reward_df, trades, {})

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "symbol": symbol,
        "frame": frame,
        "ts": dt.datetime.utcnow().isoformat(),
        **metrics_dict,
    }
    write_json(perf_dir / "summary.json", summary)

    step_val = int(reward_df["step"].iloc[-1]) if not reward_df.empty else 0
    eq_val = float(equity.iloc[-1]) if not equity.empty else 0.0
    portfolio = {
        "equity": eq_val,
        "cash": eq_val,
        "positions": {},
        "unrealized": 0.0,
        "step": step_val,
        "ts": dt.datetime.utcnow().isoformat(),
    }
    write_json(perf_dir / "portfolio_state.json", portfolio)

    try:
        fig, ax = plt.subplots()
        ax.plot(equity)
        ax.set_title("Equity Curve")
        write_png(perf_dir / "equity_curve.png", fig)

        fig, ax = plt.subplots()
        ax.plot(drawdown)
        ax.set_title("Drawdown")
        write_png(perf_dir / "drawdown.png", fig)
    except Exception:
        pass

    return summary


def main(argv: list[str] | None = None) -> int:
    ensure_headless_once("eval_run")
    p = argparse.ArgumentParser(
        description="Synthetic run evaluation",
        epilog=(
            "Example: python -m bot_trade.tools.eval_run "
            "--symbol BTCUSDT --frame 1m --run-id latest"
        ),
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--run-id")
    p.add_argument("--wfa-splits", type=int, default=0)
    p.add_argument("--wfa-embargo", type=float, default=0.01)
    p.add_argument("--tearsheet", action="store_true")
    p.add_argument("--pdf", action="store_true")
    args = p.parse_args(argv)
    run_id = args.run_id
    if not run_id or str(run_id).lower() in {"latest", "last"}:
        rid = latest_run(args.symbol, args.frame, Path(DEFAULT_REPORTS_DIR) / "PPO")
        if not rid:
            print("[LATEST] none")
            return 2
        run_id = rid
    rp = RunPaths(args.symbol, args.frame, str(run_id))

    charts_dir, img_count, rows = export_charts.export_run_charts(rp, run_id)
    print(
        "[DEBUG_EXPORT] reward_rows=%d step_rows=%d train_rows=%d risk_rows=%d callbacks_rows=%d signals_rows=%d"
        % (
            rows.get("reward", 0),
            rows.get("step", 0),
            rows.get("train", 0),
            rows.get("risk", 0),
            rows.get("callbacks", 0),
            rows.get("signals", 0),
        )
    )
    print(f"[CHARTS] dir={charts_dir.resolve()} images={img_count}")

    try:
        summary = evaluate_run(args.symbol, args.frame, run_id=run_id)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    out_dir = rp.performance_dir

    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "null"

    metrics_part = (
        "{win_rate:%s, sharpe:%s, sortino:%s, calmar:%s, max_drawdown:%s, turnover:%s, avg_trade_pnl:%s}"
        % (
            _fmt(summary.get("win_rate")),
            _fmt(summary.get("sharpe")),
            _fmt(summary.get("sortino")),
            _fmt(summary.get("calmar")),
            _fmt(summary.get("max_drawdown")),
            _fmt(summary.get("turnover")),
            _fmt(summary.get("avg_trade_pnl")),
        )
    )
    print(
        f"[EVAL] symbol={args.symbol} frame={args.frame} run_id={run_id} metrics={metrics_part}"
    )

    if args.wfa_splits:
        from bot_trade.eval.walk_forward import walk_forward_eval

        embargo = max(0.0, min(args.wfa_embargo, 0.5))
        wfa_res = walk_forward_eval(rp.logs, n_splits=args.wfa_splits, embargo=embargo)
        json_path = out_dir / "wfa_summary.json"
        write_json(json_path, wfa_res)
        csv_path = out_dir / "wfa_summary.csv"
        try:
            import pandas as pd

            df = pd.DataFrame(wfa_res.get("folds", []))
            write_text(csv_path, df.to_csv(index=False) if not df.empty else "")
        except Exception:
            write_text(csv_path, "")
        print(
            f"[WFA] run_id={run_id} splits={args.wfa_splits} embargo={embargo} out={json_path.resolve()}"
        )

    if args.tearsheet:
        from bot_trade.eval.tearsheet import generate_tearsheet

        ts_path = generate_tearsheet(rp, pdf=args.pdf)
        print(f"[TEARSHEET] out={ts_path.resolve()}")

    print(
        "[POSTRUN] run_id=%s algorithm=%s eval_win_rate=%s eval_sharpe=%s eval_max_drawdown=%s"
        % (
            run_id,
            rp.algo,
            _fmt(summary.get("win_rate")),
            _fmt(summary.get("sharpe")),
            _fmt(summary.get("max_drawdown")),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
