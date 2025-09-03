from __future__ import annotations
"""Execution bridge providing unified interface for backtest/paper/live."""
from dataclasses import dataclass
from typing import Optional

from .order_sim import OrderSimulator, Fees
from .portfolio import Portfolio


@dataclass
class ExecutionBridge:
    mode: str
    simulator: OrderSimulator
    portfolio: Portfolio

    killed: bool = False

    def submit_order(self, symbol: str, side: str, qty: float, mid: float, spread: float) -> dict:
        if self.killed:
            return {"filled_qty": 0.0, "avg_price": mid, "fees": 0.0, "partial": False, "latency_ms": 0, "rejected": True}
        fill = self.simulator.execute(side, qty, None, mid, spread)
        if not fill.get("rejected"):
            self.portfolio.update_on_fill(symbol, side, fill["filled_qty"], fill["avg_price"], fill.get("fees", 0.0))
        return fill

    def mark(self, symbol: str, price: float) -> None:
        self.portfolio.mark_to_market(symbol, price)

    def kill(self, reason: str) -> None:
        if not self.killed:
            print(f"[RISK_KILL] reason={reason}")
            self.killed = True


def build_bridge(
    mode: str,
    *,
    slippage: str = "fixed_bp",
    latency_ms: int = 0,
    partial_fills: bool = True,
    maker_bps: float = 0.0,
    taker_bps: float = 0.0,
) -> ExecutionBridge:
    sim = OrderSimulator(
        model=slippage,
        latency_ms=latency_ms,
        allow_partial=partial_fills,
        fees=Fees(maker_bps=maker_bps, taker_bps=taker_bps),
    )
    bridge = ExecutionBridge(mode=mode, simulator=sim, portfolio=Portfolio())
    if mode == "live":
        print("[LIVE_STUB]")
    return bridge
