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
        maker_fee: float | None = None,
        taker_fee: float | None = None,
        lot_size: float = 0.0,
        min_notional: float = 0.0,
        max_spread_bp: float = float("inf"),
    ) -> None:
        mf = maker_fee if maker_fee is not None else fee_bp
        tf = taker_fee if taker_fee is not None else fee_bp
        fees = Fees(maker_bps=mf, taker_bps=tf)
        super().__init__(
            model=model,
            params=params,
            latency_ms=latency_ms,
            allow_partial=allow_partial,
            fees=fees,
            min_notional=min_notional,
            lot_size=lot_size,
        )
        self.max_spread_bp = max_spread_bp
        self.fee_bp = tf

__all__ = ["ExecutionSim"]
