from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, TYPE_CHECKING

from bot_trade.eval import metrics
from bot_trade.eval.utils import load_returns, load_trades
from bot_trade.tools.atomic_io import write_json

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import numpy as np


class PurgedEmbargoSplit:
    def __init__(self, n_splits: int, embargo_frac: float) -> None:
        self.n_splits = max(1, int(n_splits))
        self.embargo_frac = float(max(0.0, embargo_frac))

    def split(self, X: 'pd.Series') -> Iterator[Tuple['np.ndarray', 'np.ndarray']]:
        import numpy as np

        n = len(X)
        if n == 0:
            return
        fold_size = max(1, n // self.n_splits)
        embargo = int(np.ceil(self.embargo_frac * n))
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min(n, test_start + fold_size)
            test_indices = np.arange(test_start, test_end)
            pre = np.arange(0, max(0, test_start - embargo))
            post = np.arange(min(n, test_end + embargo), n)
            train_indices = np.concatenate([pre, post])
            yield train_indices, test_indices


def walk_forward_eval(log_dir: Path, n_splits: int = 5, embargo: float = 0.01, period: str = "daily") -> Dict[str, Any]:
    returns = load_returns(Path(log_dir))
    trades = load_trades(Path(log_dir))
    pe = PurgedEmbargoSplit(n_splits, embargo)
    folds = []
    if len(returns) >= 2:
        for _, test_idx in pe.split(returns):
            if len(test_idx) == 0:
                continue
            test_returns = returns.iloc[test_idx]
            eq = metrics.to_equity_from_returns(test_returns, start=0.0)
            td = trades.iloc[test_idx] if not trades.empty else trades
            folds.append(
                {
                    "start": int(test_idx[0]),
                    "end": int(test_idx[-1]),
                    "sharpe": metrics.sharpe(test_returns, period=period),
                    "sortino": metrics.sortino(test_returns, period=period),
                    "calmar": metrics.calmar(eq),
                    "max_dd": metrics.max_drawdown(eq),
                    "turnover": metrics.turnover(td),
                    "win_rate": metrics.win_rate(td),
                    "avg_trade_pnl": metrics.avg_trade_pnl(td),
                }
            )
    aggregate = {
        "sharpe": metrics.sharpe(returns, period=period),
        "sortino": metrics.sortino(returns, period=period),
        "calmar": metrics.calmar(metrics.to_equity_from_returns(returns, start=0.0)),
        "max_dd": metrics.max_drawdown(metrics.to_equity_from_returns(returns, start=0.0)),
        "turnover": metrics.turnover(trades),
        "win_rate": metrics.win_rate(trades),
        "avg_trade_pnl": metrics.avg_trade_pnl(trades),
    }
    return {
        "n_splits": n_splits,
        "embargo": embargo,
        "folds": folds,
        "aggregate": aggregate,
    }


def main(argv: list[str] | None = None) -> int:
    from bot_trade.config.rl_paths import RunPaths, DEFAULT_REPORTS_DIR
    from bot_trade.tools.latest import latest_run
    from bot_trade.tools._headless import ensure_headless_once

    ensure_headless_once("eval.walk_forward")
    ap = argparse.ArgumentParser(description="Walk-forward evaluation")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ns = ap.parse_args(argv)

    run_id = ns.run_id
    if not run_id or str(run_id).lower() in {"latest", "last"}:
        rid = latest_run(ns.symbol, ns.frame, Path(DEFAULT_REPORTS_DIR) / "PPO")
        if not rid:
            print("[LATEST] none")
            return 2
        run_id = rid
    rp = RunPaths(ns.symbol, ns.frame, run_id)
    try:
        result = walk_forward_eval(rp.logs, n_splits=ns.splits, embargo=ns.embargo)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    out_path = rp.performance_dir / "wfa.json"
    write_json(out_path, result)
    print(
        f"[WFA] run_id={run_id} splits={ns.splits} embargo={ns.embargo} out={out_path.resolve()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
