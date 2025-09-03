from __future__ import annotations
"""Order simulation engine supporting multiple slippage models.

This module acts as the single source of truth for synthetic order
execution across backtest, paper and live modes.  The implementation is
lightweight but covers latency, partial fills, maker/taker fees and basic
lot/min-notional checks.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from bot_trade.tools.execution import get_slippage_model


@dataclass
class Fees:
    maker_bps: float = 0.0
    taker_bps: float = 0.0


class OrderSimulator:
    """Simple pluggable execution simulator."""

    def __init__(
        self,
        *,
        model: str = "fixed_bp",
        params: Optional[Dict[str, Any]] = None,
        latency_ms: int = 0,
        allow_partial: bool = True,
        fees: Fees | None = None,
        min_notional: float = 0.0,
        lot_size: float = 0.0,
    ) -> None:
        self.model = model.lower()
        self.params = params or {}
        self.latency_ms = int(latency_ms)
        self.allow_partial = bool(allow_partial)
        self.fees = fees or Fees()
        self.min_notional = float(min_notional)
        self.lot_size = float(lot_size)

    # ------------------------------------------------------------------
    def _slippage_bp(self, vol: Optional[float], depth: Optional[float]) -> float:
        model_fn = get_slippage_model(self.model)
        try:
            return float(model_fn(self.params, vol, depth))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _check_limits(self, qty: float, price: float) -> bool:
        if self.lot_size > 0 and qty % self.lot_size != 0:
            return False
        if self.min_notional > 0 and qty * price < self.min_notional:
            return False
        return True

    # ------------------------------------------------------------------
    def execute(
        self,
        side: str,
        qty: float,
        ts: Any,
        mid: float,
        spread: float,
        vol: Optional[float] = None,
        depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simulate execution returning fill details."""

        if not self._check_limits(qty, mid):
            return {"filled_qty": 0.0, "avg_price": mid, "fees": 0.0, "partial": False, "rejected": True}

        slippage_bp = self._slippage_bp(vol, depth)
        price_adj = mid * slippage_bp / 10_000.0
        if side.lower() == "buy":
            fill_price = mid + spread / 2.0 + price_adj
        else:
            fill_price = mid - spread / 2.0 - price_adj

        if self.allow_partial and depth is not None:
            fill_qty = min(float(qty), float(depth))
            partial = fill_qty < float(qty)
        else:
            fill_qty = float(qty)
            partial = False

        fee_bps = self.fees.maker_bps if partial else self.fees.taker_bps
        fees = fill_qty * fill_price * fee_bps / 10_000.0
        return {
            "filled_qty": fill_qty,
            "avg_price": fill_price,
            "fees": fees,
            "slippage_bp": slippage_bp,
            "partial": partial,
            "latency_ms": self.latency_ms,
            "rejected": False,
        }
