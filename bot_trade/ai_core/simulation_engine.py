# ai_core/simulation_engine.py
# Simple, fast pre-trade simulation with spread/slippage + RR check
from __future__ import annotations
from typing import Sequence


def _price_at(entry_price: float, spread: float, slippage: float, side: int) -> float:
    """Compute effective fill price after spread+slippage.
    side: +1 long, -1 short
    """
    adj = spread + slippage
    return entry_price * (1.0 + adj * side)


def simulate_entry(
    close_series: Sequence[float],
    *,
    signal_time_idx: int,
    side: int = +1,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    spread: float = 0.0,
    slippage: float = 0.0,
    lookahead: int = 100,
    min_rr: float = 1.2,
) -> int:
    """Return {-1,0,1} where 1=TP first, -1=SL first, 0=undecided.

    Assumptions:
    - Uses close-only path; fast and robust for gating.
    - side=+1 (long) by default. Short supported symmetrically.
    """
    n = len(close_series) if close_series is not None else 0
    if n == 0 or signal_time_idx is None or signal_time_idx >= n - 1:
        return 0

    entry_raw = float(close_series[signal_time_idx])
    entry = _price_at(entry_raw, spread=spread, slippage=slippage, side=+1 if side > 0 else -1)

    if side > 0:  # long
        tp = entry * (1.0 + float(tp_pct))
        sl = entry * (1.0 - float(sl_pct))
    else:  # short
        tp = entry * (1.0 - float(tp_pct))
        sl = entry * (1.0 + float(sl_pct))

    # RR check first
    rr = (abs(tp - entry)) / max(1e-9, abs(entry - sl))
    if rr < float(min_rr):
        return -1  # reject trade upfront

    end = min(n, signal_time_idx + int(lookahead) + 1)
    path = close_series[signal_time_idx + 1 : end]

    if side > 0:
        for p in path:
            if p >= tp:
                return 1
            if p <= sl:
                return -1
    else:
        for p in path:
            if p <= tp:
                return 1
            if p >= sl:
                return -1
    return 0


if __name__ == "__main__":
    import numpy as np
    prices = 100 + np.cumsum(np.random.randn(300)).astype("float32")
    print(simulate_entry(prices, signal_time_idx=150))
