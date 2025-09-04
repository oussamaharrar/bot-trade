from __future__ import annotations
"""Binance Testnet adapter used by :class:`SandboxGateway`.

Only a very small subset of Binance's REST API is implemented.  All requests
are directed to the public testnet host and require API keys provided via the
``BINANCE_API_KEY`` and ``BINANCE_API_SECRET`` environment variables.
"""

from dataclasses import dataclass
import hashlib
import hmac
import os
import time
from typing import Dict, Optional

import requests

from bot_trade.utils.rate_limit import RateLimiter, retry
from bot_trade.gateways.errors import GatewayError


@dataclass
class BinanceTestnet:
    base_url: str
    recv_window: int
    limiter: RateLimiter
    routes: Dict[str, int]
    time_sync: bool = False
    session: requests.Session = requests.Session()

    def __post_init__(self) -> None:
        self.key = os.environ.get("BINANCE_API_KEY")
        self.secret = os.environ.get("BINANCE_API_SECRET")
        if not self.key or not self.secret or len(self.key) < 5 or len(self.secret) < 5:
            raise GatewayError(0, "Missing or malformed BINANCE_API_KEY/SECRET")
        self.time_offset = 0.0
        if self.time_sync:
            self._sync_time()

    def _sync_time(self) -> None:
        try:
            r = self.session.get(f"{self.base_url}/api/v3/time", timeout=5)
            srv = int(r.json().get("serverTime", 0))
            self.time_offset = (srv - int(time.time() * 1000)) / 1000.0
        except Exception as exc:  # pragma: no cover - network
            raise GatewayError(0, f"time sync failed: {exc}")

    # ------------------------------------------------------------------
    def _sign(self, params: Dict[str, str]) -> Dict[str, str]:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        sig = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _request(self, method: str, path: str, params: Optional[Dict[str, str]] = None,
                 weight: int = 1, auth: bool = False) -> Dict[str, any]:
        weight = self.routes.get(path, weight)
        self.limiter.acquire(path, weight)
        params = params or {}
        params["recvWindow"] = str(self.recv_window)
        params["timestamp"] = str(int((time.time() + self.time_offset) * 1000))
        if auth:
            params = self._sign(params)
        url = f"{self.base_url}{path}"

        def send() -> Dict[str, any]:
            r = self.session.request(method, url, params=params, headers={"X-MBX-APIKEY": self.key})
            return {"status": r.status_code, "json": r.json() if r.content else {}}

        resp = retry(
            send,
            is_retryable=lambda r: r["status"] in {418, 429, 1003},
            route=path,
            weight=weight,
        )
        status = resp["status"]
        if status != 200:
            raise GatewayError(status, str(resp["json"]))
        return resp["json"]

    # ------------------------------------------------------------------
    def get_price(self, symbol: str) -> float:
        data = self._request("GET", "/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    def get_balance(self, asset: str) -> float:
        data = self._request("GET", "/api/v3/account", auth=True)
        for bal in data.get("balances", []):
            if bal.get("asset") == asset:
                return float(bal.get("free", 0))
        return 0.0

    def place_order(self, symbol: str, side: str, qty: float, order_type: str,
                    price: Optional[float] = None) -> Dict[str, any]:
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(qty),
        }
        if price is not None:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"
        return self._request("POST", "/api/v3/order", params, weight=1, auth=True)

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        params = {"symbol": symbol, "orderId": str(order_id)}
        self._request("DELETE", "/api/v3/order", params, weight=1, auth=True)
        return True
