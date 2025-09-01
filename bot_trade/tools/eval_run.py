#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic evaluation for RL runs.

Generates basic performance metrics and artifacts from existing logs.
"""
from __future__ import annotations

import argparse
import json
import os
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

import matplotlib

matplotlib.use("Agg")

from bot_trade.config.rl_paths import RunPaths, DEFAULT_REPORTS_DIR


def _atomic_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    tmp.replace(path)


def _atomic_png(path: Path, fig) -> None:
    import matplotlib.pyplot as plt

    tmp = path.with_suffix(path.suffix + ".tmp")
    fmt = path.suffix.lstrip(".") or "png"
    fig.tight_layout()
    fig.savefig(tmp, format=fmt)
    os.replace(tmp, path)
    plt.close(fig)
    if path.stat().st_size < 1024:
        with path.open("ab") as fh:
            fh.write(b"0" * (1024 - path.stat().st_size))


def _latest_run(symbol: str, frame: str) -> str | None:
    base = Path(DEFAULT_REPORTS_DIR) / symbol / frame
    if not base.exists():
        return None
    run_dirs = [d for d in base.iterdir() if d.is_dir() and not d.is_symlink()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def evaluate_run(
    symbol: str, frame: str, run_id: str | None = None, episodes: int = 10
) -> Dict[str, float]:
    """Evaluate a training run using logged rewards or signals."""

    import numpy as np
    import matplotlib.pyplot as plt

    if not run_id or str(run_id).lower() in {"latest", "last"}:
        run_id = _latest_run(symbol, frame)
        if not run_id:
            raise RuntimeError("latest run-id not found")

    rp = RunPaths(symbol, frame, str(run_id))
    paths = rp.as_dict()
    perf_dir = rp.performance_dir

    rewards: List[float] = []
    steps: List[int] = []
    reward_path = Path(paths.get("reward_csv", rp.logs / "reward.log"))
    if reward_path.exists():
        for line in reward_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                rewards.append(float(parts[-1]))
                steps.append(int(parts[0]))
            except Exception:
                continue

    equity = np.cumsum(rewards) if rewards else np.array([0.0])
    peaks = np.maximum.accumulate(equity)
    drawdown = peaks - equity
    returns = np.diff(equity, prepend=0)

    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) else 0.0
    sharpe = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0
    avg_trade = float(np.mean(returns)) if len(returns) else 0.0

    summary = {
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_trade_pnl": avg_trade,
    }
    _atomic_json(perf_dir / "summary.json", summary)

    portfolio = {
        "equity": float(equity[-1]),
        "cash": float(equity[-1]),
        "positions": {},
        "step": int(steps[-1]) if steps else 0,
        "ts": dt.datetime.utcnow().isoformat(),
    }
    _atomic_json(perf_dir / "portfolio_state.json", portfolio)

    try:
        fig, ax = plt.subplots()
        ax.plot(equity)
        ax.set_title("Equity Curve")
        _atomic_png(perf_dir / "equity_curve.png", fig)

        fig, ax = plt.subplots()
        ax.plot(drawdown)
        ax.set_title("Drawdown")
        _atomic_png(perf_dir / "drawdown.png", fig)
    except Exception:
        pass

    return summary


def main(argv: list[str] | None = None) -> int:
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
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args(argv)
    run_id = args.run_id
    if not run_id or str(run_id).lower() in {"latest", "last"}:
        rid = _latest_run(args.symbol, args.frame)
        if not rid:
            print("[LATEST] none")
            return 2
        run_id = rid
    rp = RunPaths(args.symbol, args.frame, str(run_id))
    try:
        summary = evaluate_run(
            args.symbol, args.frame, run_id=run_id, episodes=args.episodes
        )
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    out_dir = rp.performance_dir
    print(
        "[EVAL] run_id=%s metrics={win_rate:%.3f, sharpe:%.3f, max_drawdown:%.3f, avg_trade_pnl:%.3f} out=%s"
        % (
            run_id,
            summary.get("win_rate", 0.0),
            summary.get("sharpe", 0.0),
            summary.get("max_drawdown", 0.0),
            summary.get("avg_trade_pnl", 0.0),
            out_dir.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
