#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthetic evaluation for RL runs.

Generates basic performance metrics and artifacts from existing logs.
"""
from __future__ import annotations

import argparse
import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot_trade.config.rl_paths import RunPaths, memory_dir


def _atomic_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    tmp.replace(path)


def _latest_run_id() -> str | None:
    state = memory_dir() / "state_latest.json"
    if not state.exists():
        return None
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
        rid = data.get("last_run_id")
        return str(rid) if rid else None
    except Exception:
        return None


def evaluate_run(symbol: str, frame: str, run_id: str | None = None, episodes: int = 10) -> Dict[str, float]:
    """Evaluate a training run using logged rewards or signals."""
    if not run_id or str(run_id).lower() in {"latest", "last"}:
        run_id = _latest_run_id()
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
        plt.figure()
        plt.plot(equity)
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig(perf_dir / "equity_curve.png")
        plt.close()

        plt.figure()
        plt.plot(drawdown)
        plt.title("Drawdown")
        plt.tight_layout()
        plt.savefig(perf_dir / "drawdown.png")
        plt.close()
    except Exception:
        pass

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic run evaluation")
    p.add_argument("--symbol", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--run-id")
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()
    evaluate_run(args.symbol, args.frame, run_id=args.run_id, episodes=args.episodes)


if __name__ == "__main__":
    main()
