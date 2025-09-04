from __future__ import annotations

"""Thin wrapper exposing the paper gateway under the live namespace."""

from typing import Any, Dict

import pandas as pd

from bot_trade.gateways.paper import PaperGateway as _PaperGateway
from .gateway_base import GatewayBase


class PaperGateway(_PaperGateway, GatewayBase):
    def place_order(
        self,
        side: str,
        qty: float,
        price: float | None,
        otype: str = "market",
        client_order_id: str | None = None,
    ) -> Dict[str, Any]:
        order = {"side": side, "qty": qty, "price": price, "type": otype}
        if client_order_id:
            order["client_order_id"] = client_order_id
        return self.submit(order)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return {"id": order_id, "cancelled": bool(self.cancel(order_id))}

    def fetch_positions(self) -> pd.DataFrame:
        return pd.DataFrame(self.positions(), index=[0])

    def fetch_balance(self) -> Dict[str, Any]:
        return self.balances()

    def fetch_trades(self, since=None, limit: int = 1000) -> pd.DataFrame:
        return pd.DataFrame(self.poll_fills())


__all__ = ["PaperGateway"]
