from __future__ import annotations

"""Helpers for building equity and drawdown series."""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from .utils import load_returns
from .metrics import equity_from_rewards


def build_equity_drawdown(
    log_dir: Path,
    cfg: Optional[Dict[str, Any]] = None,
    window: Optional[int] = None,
    fill_value: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:
    """Return equity and drawdown series from ``log_dir``.

    ``window`` limits the number of return rows considered. Missing values
    are filled using ``fill_value``. ``cfg`` may include a starting equity via
    ``{"start": float}"`.
    """

    returns = load_returns(Path(log_dir))
    if window is not None:
        returns = returns.tail(int(window))
    returns = returns.fillna(fill_value)
    rewards_df = returns.to_frame(name="reward")
    equity = equity_from_rewards(rewards_df, cfg)
    drawdown = (
        equity / equity.cummax() - 1 if not equity.empty else pd.Series(dtype=float)
    )
    return equity, drawdown
