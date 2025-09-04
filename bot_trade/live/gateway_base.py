from __future__ import annotations

"""Abstract gateway interface for execution backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd


class GatewayBase(ABC):
    @abstractmethod
    def place_order(
        self,
        side: str,
        qty: float,
        price: float | None,
        otype: str = "market",
        client_order_id: str | None = None,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def fetch_positions(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def fetch_balance(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def fetch_trades(self, since=None, limit: int = 1000) -> pd.DataFrame:
        ...
