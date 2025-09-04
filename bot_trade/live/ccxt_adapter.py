from __future__ import annotations

"""Re-export of existing ccxt adapter under live namespace."""

from typing import Any, Dict

import pandas as pd

from bot_trade.gateways.ccxt_adapter import CCXTAdapter as _CCXTAdapter
from .gateway_base import GatewayBase


class CCXTAdapter(_CCXTAdapter, GatewayBase):
    def cancel_order(self, order_id: str) -> Dict[str, Any]:  # pragma: no cover - dry run
        return {"id": order_id, "status": "cancelled"}

    def fetch_positions(self) -> pd.DataFrame:  # pragma: no cover - dry run
        return pd.DataFrame()

    def fetch_balance(self) -> Dict[str, Any]:  # pragma: no cover - dry run
        return {}

    def fetch_trades(self, since=None, limit: int = 1000) -> pd.DataFrame:  # pragma: no cover
        return pd.DataFrame()


__all__ = ["CCXTAdapter"]
