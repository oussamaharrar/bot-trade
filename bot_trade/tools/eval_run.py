#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic evaluation for RL runs.

Generates basic performance metrics and artifacts from existing logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

from bot_trade.config.rl_paths import RunPaths, DEFAULT_REPORTS_DIR
from bot_trade.tools.atomic_io import write_json, write_png, write_text
from bot_trade.tools.latest import latest_run
from bot_trade.tools import export_charts
from bot_trade.tools.kb_writer import kb_append
from bot_trade.eval.gate import gate_metrics
from bot_trade.tools.force_utf8 import force_utf8
from bot_trade.tools._headless import ensure_headless_once




def _evaluate_run(symbol: str, frame: str, run_id: str, algo: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute evaluation metrics and equity/drawdown series."""

    import pandas as pd
    import matplotlib.pyplot as plt
    from bot_trade.eval import metrics
    from bot_trade.eval.utils import load_reward_log, load_trades
    from bot_trade.eval.equity import build_equity_drawdown

    rp = RunPaths(symbol, frame, str(run_id), algo)
    perf_dir = rp.performance_dir

    reward_df = load_reward_log(rp.logs)
    trades = load_trades(rp.logs)
    equity, drawdown = build_equity_drawdown(rp.logs)
    metrics_dict = metrics.compute_all(equity.to_frame(name="equity"), trades, {})

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "symbol": symbol,
        "frame": frame,
        "ts": dt.datetime.utcnow().isoformat(),
        **metrics_dict,
        "equity_len": int(len(equity)),
    }
    write_json(perf_dir / "summary.json", summary)
    try:
        with (perf_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=metrics_dict.keys())
            w.writeheader()
            w.writerow(metrics_dict)
    except Exception:
        pass

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

    return summary, portfolio


def evaluate_run(symbol: str, frame: str, run_id: str, algo: str = "PPO") -> Dict[str, Any]:
    """Compatibility wrapper returning only the summary."""

    summary, _ = _evaluate_run(symbol, frame, run_id, algo)
    return summary


def main(argv: list[str] | None = None) -> int:
    force_utf8()

    ensure_headless_once("tools.eval_run")
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
    p.add_argument("--algorithm", default="PPO")
    p.add_argument("--wfa", action="store_true", help="Run walk-forward with default splits")
    p.add_argument("--wfa-splits", type=int, default=0)
    p.add_argument("--wfa-embargo", type=float, default=0.01)
    p.add_argument("--tearsheet", action="store_true")
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--gate", action="store_true")
    args = p.parse_args(argv)
    run_id = args.run_id
    algo = args.algorithm.upper()
    if not run_id or str(run_id).lower() in {"latest", "last"}:
        rid = latest_run(args.symbol, args.frame, Path(DEFAULT_REPORTS_DIR) / algo)
        if not rid:
            print("[LATEST] none")
            return 2
        run_id = rid
    rp = RunPaths(args.symbol, args.frame, str(run_id), algo)

    charts_dir, img_count, rows = export_charts.export_run_charts(rp, run_id)
    required = {"reward.png", "sharpe.png", "loss.png", "entropy.png", "risk_flags.png"}
    for name in required:
        target = charts_dir / name
        if not target.exists():
            try:
                export_charts._placeholder(target, "NO DATA")
            except Exception:
                pass
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
        summary, portfolio = _evaluate_run(args.symbol, args.frame, run_id=run_id, algo=algo)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    out_dir = rp.performance_dir

    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "null"

    metrics_part = (
        "{sharpe:%s, sortino:%s, calmar:%s, max_drawdown:%s, turnover:%s, win_rate:%s, avg_trade_pnl:%s}"
        % (
            _fmt(summary.get("sharpe")),
            _fmt(summary.get("sortino")),
            _fmt(summary.get("calmar")),
            _fmt(summary.get("max_drawdown")),
            _fmt(summary.get("turnover")),
            _fmt(summary.get("win_rate")),
            _fmt(summary.get("avg_trade_pnl")),
        )
    )
    print(
        f"[EVAL] symbol={args.symbol} frame={args.frame} run_id={run_id} metrics={metrics_part}"
    )

    splits = args.wfa_splits if not args.wfa else max(args.wfa_splits, 5)
    wfa_pass = True
    if splits:
        from bot_trade.eval.walk_forward import walk_forward_eval

        embargo = max(0.0, min(args.wfa_embargo, 0.5))
        wfa_res = walk_forward_eval(rp.logs, n_splits=splits, embargo=embargo)
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
        wfa_pass = bool(wfa_res.get("folds"))

    if args.tearsheet:
        from bot_trade.eval.tearsheet import generate_tearsheet

        generate_tearsheet(rp, pdf=args.pdf)

    images_list = sorted(
        [p.name for p in charts_dir.glob("*.png") if p.is_file() and not p.is_symlink()]
    )
    gate_pass, gate_details = gate_metrics(
        {
            "sharpe": summary.get("sharpe"),
            "max_drawdown": summary.get("max_drawdown"),
            "win_rate": summary.get("win_rate"),
            "turnover": summary.get("turnover"),
        },
        wfa_pass=wfa_pass,
    )
    print(
        f"[GATE] pass={str(gate_pass).lower()} details=[{','.join(gate_details)}]"
    )

    kb_payload = {
        "run_id": run_id,
        "symbol": args.symbol,
        "frame": args.frame,
        "algorithm": rp.algo,
        "ts": dt.datetime.utcnow().isoformat(),
        "images": len(images_list),
        "images_list": images_list,
        "rows_reward": rows.get("reward", 0),
        "rows_step": rows.get("step", 0),
        "rows_train": rows.get("train", 0),
        "rows_risk": rows.get("risk", 0),
        "rows_callbacks": rows.get("callbacks", 0),
        "rows_signals": rows.get("signals", 0),
        "vecnorm_applied": rp.features.vecnorm.exists(),
        "eval": {
            "sharpe": summary.get("sharpe"),
            "sortino": summary.get("sortino"),
            "calmar": summary.get("calmar"),
            "max_drawdown": summary.get("max_drawdown"),
            "turnover": summary.get("turnover"),
            "win_rate": summary.get("win_rate"),
            "avg_trade_pnl": summary.get("avg_trade_pnl"),
        },
        "gate_pass": gate_pass,
        "gate_details": gate_details,
        "portfolio": portfolio,
    }
    reg_file = rp.performance_dir / "adaptive_log.jsonl"
    if reg_file.exists():
        try:
            import json as _json
            from collections import Counter

            counts: Counter[str] = Counter()
            active = ""
            with reg_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    regime = rec.get("regime")
                    if regime is None:
                        continue
                    regime = str(regime)
                    counts[regime] += 1
                    active = regime
            total = sum(counts.values())
            dist = {k: v / total for k, v in counts.items()} if total else {}
            kb_payload["regime"] = {"active": active, "distribution": dist}
        except Exception:
            pass

    kb_append(rp, kb_payload)

    print(
        "[POSTRUN] run_id=%s algorithm=%s eval_win_rate=%s eval_sharpe=%s eval_max_drawdown=%s gate_pass=%s"
        % (
            run_id,
            rp.algo,
            _fmt(summary.get("win_rate")),
            _fmt(summary.get("sharpe")),
            _fmt(summary.get("max_drawdown")),
            str(gate_pass).lower(),
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
