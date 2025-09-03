from __future__ import annotations
"""Minimal portfolio tracker used by the unified execution bridge."""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class Portfolio:
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    exposure: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    turnover: float = 0.0

    # ------------------------------------------------------------------
    def update_on_fill(self, symbol: str, side: str, qty: float, price: float, fees: float) -> None:
        pos = self.positions.setdefault(symbol, Position())
        if side.lower() == "buy":
            total_cost = pos.avg_price * pos.qty + qty * price + fees
            pos.qty += qty
            pos.avg_price = total_cost / pos.qty if pos.qty else 0.0
            self.cash -= qty * price + fees
        else:
            pnl = (price - pos.avg_price) * qty - fees
            pos.qty -= qty
            self.cash += qty * price - fees
            self.realized_pnl += pnl
            self.turnover += abs(qty * price)
            if pos.qty == 0:
                pos.avg_price = 0.0

    # ------------------------------------------------------------------
    def mark_to_market(self, symbol: str, price: float) -> None:
        pos = self.positions.get(symbol)
        if not pos:
            self.equity = self.cash
            self.unrealized_pnl = 0.0
            self.exposure = 0.0
            return
        self.unrealized_pnl = (price - pos.avg_price) * pos.qty
        self.equity = self.cash + pos.qty * price
        notional = abs(pos.qty * price)
        self.exposure = notional / self.equity if self.equity else 0.0
