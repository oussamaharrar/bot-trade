from __future__ import annotations
"""CCXT-based exchange adapter stub.

This adapter normalises symbols and provides basic ticker/order book
fetching.  No API keys are required; all methods are dry-run.  A single
[GATEWAY] line is printed upon initialisation.
"""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CCXTAdapter:
    symbol: str
    sandbox: bool = True

    def __post_init__(self) -> None:
        print(f"[GATEWAY] provider=ccxt sandbox={self.sandbox}")

    # ------------------------------------------------------------------
    def fetch_ticker(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "bid": 0.0, "ask": 0.0}

    def fetch_order_book(self) -> Dict[str, Any]:
        return {"bids": [], "asks": []}

    def place_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dry-run order placement returning echo payload."""
        return {"status": "dry-run", "args": args, "kwargs": kwargs}
