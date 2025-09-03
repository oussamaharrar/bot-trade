from __future__ import annotations
"""Deprecated shim mapping to :mod:`bot_trade.env.execution.order_sim`."""
from .execution.order_sim import OrderSimulator, Fees

print("[DEPRECATION] use bot_trade.env.execution.order_sim.OrderSimulator")


class ExecutionSim(OrderSimulator):
    def __init__(
        self,
        *,
        model: str,
        params=None,
        latency_ms: int = 0,
        allow_partial: bool = True,
        fee_bp: float = 0.0,
        max_spread_bp: float = float("inf"),
    ) -> None:
        fees = Fees(maker_bps=fee_bp, taker_bps=fee_bp)
        super().__init__(model=model, params=params, latency_ms=latency_ms, allow_partial=allow_partial, fees=fees)
        self.max_spread_bp = max_spread_bp
        self.fee_bp = fee_bp

__all__ = ["ExecutionSim"]
