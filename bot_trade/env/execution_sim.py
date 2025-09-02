from __future__ import annotations

"""Simple pluggable execution simulator supporting basic slippage models."""

from typing import Any, Dict, Optional


class ExecutionSim:
    def __init__(
        self,
        *,
        model: str,
        params: Optional[Dict[str, Any]],
        latency_ms: int,
        allow_partial: bool,
        fee_bp: float = 0.0,
    ) -> None:
        self.model = (model or "fixed_bp").lower()
        self.params = params or {}
        self.latency_ms = int(latency_ms)
        self.allow_partial = bool(allow_partial)
        self.fee_bp = float(fee_bp)

    # ------------------------------------------------------------------
    def _slippage_bp(self, vol: Optional[float], depth: Optional[float]) -> float:
        if self.model == "fixed_bp":
            return float(self.params.get("bp", 0.0))
        if self.model == "vol_aware":
            k = float(self.params.get("k", 0.0))
            return float(k * float(vol or 0.0))
        if self.model == "depth_aware":
            impact = float(self.params.get("book_impact", 0.0))
            d = float(depth or 0.0)
            if d <= 0:
                return 0.0
            return float(impact / d)
        return 0.0

    # ------------------------------------------------------------------
    def quote(self, ts, mid: float, spread: float, vol: Optional[float] = None, depth: Optional[float] = None) -> Dict[str, float]:
        slippage_bp = self._slippage_bp(vol, depth)
        bid = mid - spread / 2.0
        ask = mid + spread / 2.0
        return {"bid": bid, "ask": ask, "slippage_bp": slippage_bp}

    # ------------------------------------------------------------------
    def execute(
        self,
        side: str,
        qty: float,
        ts,
        mid: float,
        spread: float,
        vol: Optional[float] = None,
        depth: Optional[float] = None,
    ) -> Dict[str, Any]:
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

        fees = fill_qty * fill_price * self.fee_bp / 10_000.0
        return {
            "filled_qty": fill_qty,
            "avg_price": fill_price,
            "fees": fees,
            "slippage_bp": slippage_bp,
            "partial": partial,
            "latency_ms": self.latency_ms,
        }
