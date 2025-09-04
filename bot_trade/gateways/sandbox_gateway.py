from __future__ import annotations
"""Sandbox gateway wrapping Binance and Bybit testnets."""

from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional
import yaml

from bot_trade.utils.rate_limit import RateLimiter
from .exchanges.binance_testnet import BinanceTestnet, GatewayError as BinanceError
from .exchanges.bybit_testnet import BybitTestnet, GatewayError as BybitError


@dataclass
class ExecutionResult:
    status: str
    filled_qty: float
    avg_price: float
    fees: float
    id: str


class SandboxGateway:
    """Unified interface for exchange testnets."""

    def __init__(self, exchange: str, symbol: str,
                 config_path: str = "config/exchange_sandbox.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if exchange not in cfg:
            raise ValueError(f"Unsupported exchange: {exchange}")
        ecfg = cfg[exchange]
        limiter = RateLimiter(capacity=ecfg.get("weight_limit", 10))
        if exchange == "binance":
            self.adapter = BinanceTestnet(ecfg["rest"], ecfg.get("recv_window", 5000), limiter)
        elif exchange == "bybit":
            self.adapter = BybitTestnet(ecfg["rest"], limiter, ecfg.get("recv_window", 5000))
        else:
            raise ValueError(exchange)
        self.symbol = symbol
        self.exchange = exchange
        print(f"[GATEWAY] provider=sandbox exchange={exchange}")

    # ------------------------------------------------------------------
    def place_order(self, symbol: Optional[str], side: str, qty: float,
                    order_type: str, price: Optional[float] = None) -> ExecutionResult:
        symbol = symbol or self.symbol
        data = self.adapter.place_order(symbol, side, qty, order_type, price)
        return ExecutionResult(
            status=data.get("status", "UNKNOWN"),
            filled_qty=float(data.get("executedQty", 0) or data.get("qty", 0)),
            avg_price=float(data.get("price", 0) or data.get("avgPrice", 0)),
            fees=float(data.get("commission", 0)),
            id=str(data.get("orderId", data.get("order_id", ""))),
        )

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        symbol = symbol or self.symbol
        return self.adapter.cancel_order(symbol, order_id)

    def get_balance(self, asset: str) -> float:
        return self.adapter.get_balance(asset)

    def get_price(self, symbol: Optional[str] = None) -> float:
        symbol = symbol or self.symbol
        return self.adapter.get_price(symbol)

    def stream_prices(self, symbol: str, on_tick: Callable[[float], None], interval: float = 1.0) -> None:
        def _run() -> None:
            while True:
                price = self.get_price(symbol)
                on_tick(price)
                time.sleep(interval)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

__all__ = ["SandboxGateway", "ExecutionResult"]

