from __future__ import annotations
"""In-memory paper trading gateway used for smoke tests.

This gateway is intentionally lightweight: it implements the minimal
order/position interface expected by the execution bridge while avoiding any
external side effects.  All writes and fills are kept in memory and the
interface is synchronous.
"""
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List

from bot_trade.tools.force_utf8 import force_utf8


@dataclass
class PaperGateway:
    """Simple paper gateway with basic rate limiting and idempotency."""

    symbol: str
    rate_limit_per_sec: int = 5
    _orders: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    _fills: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _last_refill: float = field(default_factory=time.time, init=False)
    _tokens: int = field(init=False)

    def __post_init__(self) -> None:
        force_utf8()
        self._tokens = self.rate_limit_per_sec
        print("[GATEWAY] provider=paper")

    # ------------------------------------------------------------------
    def _refill(self) -> None:
        now = time.time()
        if now - self._last_refill >= 1.0:
            self._tokens = self.rate_limit_per_sec
            self._last_refill = now

    def _consume(self) -> bool:
        self._refill()
        if self._tokens <= 0:
            print("[GATEWAY] rate_limit exceeded")
            return False
        self._tokens -= 1
        return True

    def _deterministic_id(self, order: Dict[str, Any]) -> str:
        base = f"{order.get('side')}:{order.get('qty')}:{order.get('price')}"
        return str(abs(hash(base)) % (10 ** 12))

    # ------------------------------------------------------------------
    def submit(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Accept an order returning an acknowledgement."""
        self._consume()
        oid = self._deterministic_id(order)
        if oid in self._orders:
            return {"id": oid, "status": "duplicate"}
        self._orders[oid] = order.copy()
        return {"id": oid, "status": "accepted"}

    def poll_fills(self) -> List[Dict[str, Any]]:
        fills = self._fills[:]
        self._fills.clear()
        return fills

    def positions(self) -> Dict[str, float]:
        return {}

    def balances(self) -> Dict[str, float]:
        return {}

    def cancel(self, oid: str) -> bool:
        return self._orders.pop(oid, None) is not None

    def clock(self) -> float:
        return time.time()


__all__ = ["PaperGateway"]
