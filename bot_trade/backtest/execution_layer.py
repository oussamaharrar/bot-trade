"""Execution layer with basic slippage, fees, latency and partial fills.

This module provides a deterministic execution simulator used by backtests and
unit tests.  The design is intentionally small but mirrors the interface used
in the larger project so that environment and live runners can share the same
logic.
"""

import random
from typing import Any

from .models import ExecutionResult, Fill, Order


class ExecutionLayer:
    """Simulate order execution with configurable models.

    Parameters
    ----------
    cfg: dict
        Configuration dictionary.  Supported keys:
        ``slippage`` -> {"model": "fixed_bps"|"atr_proportional"|"depth_aware",
        parameters}
        ``fees`` -> {"maker_bps", "taker_bps", "min_fee"}
        ``latency_ms`` -> integer milliseconds delay
        ``partial_fills`` -> {"fill_ratio"}
    seed: int | None
        Seed for deterministic behaviour.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or {}
        self.rng = random.Random(seed)
        self.seed = seed

    # ------------------------------------------------------------------
    def _slippage_bps(self, order: Order, market: dict[str, Any]) -> float:
        model = self.cfg.get("slippage", {})
        mtype = model.get("model", "fixed_bps")
        price = market.get("price", order.price)
        if mtype == "fixed_bps":
            return float(model.get("bps", 0.0))
        if mtype == "atr_proportional":
            atr = float(market.get("atr", 0.0))
            mult = float(model.get("mult", 1.0))
            if price <= 0:
                return 0.0
            return (atr / price) * mult * 10_000
        if mtype == "depth_aware":
            depth = float(market.get("depth", order.qty))
            impact = order.qty / depth if depth > 0 else 0.0
            bps_per_impact = float(model.get("bps_per_impact", 1.0))
            return impact * bps_per_impact * 10_000
        return 0.0

    def _fee(self, price: float, qty: float, is_maker: bool) -> float:
        fees_cfg = self.cfg.get("fees", {})
        key = "maker_bps" if is_maker else "taker_bps"
        bps = float(fees_cfg.get(key, 0.0))
        fee = abs(price * qty) * bps / 10_000
        min_fee = float(fees_cfg.get("min_fee", 0.0))
        if fee > 0:
            fee = max(fee, min_fee)
        return fee

    # ------------------------------------------------------------------
    def apply(self, order: Order, market: dict[str, Any]) -> ExecutionResult:
        """Execute ``order`` against ``market`` and return :class:`ExecutionResult`.

        ``market`` is a mapping providing at least a ``price`` entry and
        optionally ``atr`` or ``depth`` depending on the slippage model.
        """

        price = float(market.get("price", order.price))
        slippage_bps = self._slippage_bps(order, market)
        slip_factor = 1 + slippage_bps / 10_000 if order.side == "buy" else 1 - slippage_bps / 10_000
        fill_price = price * slip_factor

        partial_cfg = self.cfg.get("partial_fills", {})
        ratio = float(partial_cfg.get("fill_ratio", 1.0))
        ratio = max(0.0, min(1.0, ratio))
        filled_qty = order.qty * ratio
        status = "filled" if ratio >= 1.0 - 1e-12 else "partial"

        fee = self._fee(fill_price, filled_qty, order.is_maker)

        latency_ms = int(self.cfg.get("latency_ms", 0))
        ts = order.ts + latency_ms / 1000.0

        fill = Fill(
            side=order.side,
            qty=filled_qty,
            price=fill_price,
            fee=fee,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
        )

        print(
            "FILL",
            f"side={order.side}",
            f"qty={filled_qty}",
            f"px={fill_price}",
            f"fee={fee}",
            f"slip_bps={slippage_bps}",
            f"latency_ms={latency_ms}",
        )

        return ExecutionResult(
            status=status,
            filled_qty=filled_qty,
            avg_price=fill_price,
            fees=fee,
            ts=ts,
            slippage_bps=slippage_bps,
            order_id=order.id,
            fills=[fill],
        )
