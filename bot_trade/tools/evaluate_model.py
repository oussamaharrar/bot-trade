"""Light‑weight evaluation helper.

The real project performs a detailed evaluation of the trained model.  For
development and CI purposes we provide a much simpler stand‑in that
generates synthetic performance metrics so downstream reporting can be
tested without heavy dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bot_trade.config.rl_paths import RunPaths


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
    os.replace(tmp, path)


def evaluate_for_run(run_paths: RunPaths, episodes: int = 3, debug: bool = False) -> Dict[str, Any]:
    """Run a synthetic evaluation for ``run_paths``.

    A set of random returns is generated for each episode.  The resulting
    metrics are saved under ``reports/<run>/performance``.
    """

    perf_dir = run_paths.performance_dir
    perf_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    records = []
    for ep in range(episodes):
        returns = rng.normal(0, 1, 100)
        total = float(returns.sum())
        win_rate = float((returns > 0).mean())
        max_dd = float(np.max(np.maximum.accumulate(returns.cumsum()) - returns.cumsum()))
        sharpe = float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(len(returns)))
        records.append(
            {
                "episode": ep + 1,
                "return": total,
                "win_rate": win_rate,
                "max_dd": max_dd,
                "sharpe": sharpe,
            }
        )

    df = pd.DataFrame(records)
    perf_csv = perf_dir / "performance.csv"
    _atomic_write(perf_csv, df.to_csv(index=False))
    _atomic_write(perf_dir / "evaluation_train.csv", df.to_csv(index=False))

    summary = {
        "mean_return": float(df["return"].mean()),
        "win_rate": float(df["win_rate"].mean()),
        "max_dd": float(df["max_dd"].max()),
        "sharpe": float(df["sharpe"].mean()),
    }
    _atomic_write(perf_dir / "perf_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

    # Charts
    fig, ax = plt.subplots()
    ax.plot(df["episode"], df["return"], marker="o")
    ax.set_title("evaluation returns")
    fig.savefig(perf_dir / "eval_returns.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(df["episode"], df["max_dd"], marker="o")
    ax.set_title("drawdown")
    fig.savefig(perf_dir / "drawdown.png")
    plt.close(fig)

    if debug:
        print(f"[EVAL] episodes={episodes} mean_return={summary['mean_return']:.3f}")

    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI helper
    import argparse
    from bot_trade.config.rl_paths import get_root

    ap = argparse.ArgumentParser("evaluate model")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base", default=None)
    ns = ap.parse_args(argv)

    root = Path(ns.base) if ns.base else get_root()
    rp = RunPaths(ns.symbol, ns.frame, ns.run_id, root=root)
    evaluate_for_run(rp, episodes=ns.episodes)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

