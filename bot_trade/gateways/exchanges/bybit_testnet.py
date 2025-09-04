from __future__ import annotations
"""Bybit Testnet adapter for :class:`SandboxGateway`."""

from dataclasses import dataclass
import hashlib
import hmac
import os
import time
from typing import Dict, Optional

import requests

from bot_trade.utils.rate_limit import RateLimiter, retry
from bot_trade.gateways.errors import GatewayError


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


@dataclass
class BybitTestnet:
    base_url: str
    limiter: RateLimiter
    recv_window: int
    routes: Dict[str, int]
    time_sync: bool = False
    session: requests.Session = requests.Session()

    def __post_init__(self) -> None:
        self.key = os.environ.get("BYBIT_API_KEY")
        self.secret = os.environ.get("BYBIT_API_SECRET")
        if not self.key or not self.secret or len(self.key) < 5 or len(self.secret) < 5:
            raise GatewayError(0, "Missing or malformed BYBIT_API_KEY/SECRET")
        self.time_offset = 0.0
        if self.time_sync:
            self._sync_time()

    def _sync_time(self) -> None:
        try:
            r = self.session.get(f"{self.base_url}/v5/market/time", timeout=5)
            srv = int(r.json().get("timeSecond", 0)) * 1000
            self.time_offset = (srv - int(time.time() * 1000)) / 1000.0
        except Exception as exc:  # pragma: no cover - network
            raise GatewayError(0, f"time sync failed: {exc}")

    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, params: Optional[Dict[str, str]] = None,
                 weight: int = 1, auth: bool = False) -> Dict[str, any]:
        weight = self.routes.get(path, weight)
        self.limiter.acquire(path, weight)
        params = params or {}
        params["timestamp"] = str(int((time.time() + self.time_offset) * 1000))
        params["recv_window"] = str(self.recv_window)
        headers = {"X-BAPI-API-KEY": self.key}
        if auth:
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            headers["X-BAPI-SIGN"] = _sign(self.secret, query)
        url = f"{self.base_url}{path}"

        def send() -> Dict[str, any]:
            r = self.session.request(method, url, params=params, headers=headers)
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
        data = self._request("GET", "/v5/market/tickers", {"symbol": symbol})
        tick = data.get("result", {}).get("list", [{}])[0]
        return float(tick.get("lastPrice", 0))

    def get_balance(self, asset: str) -> float:
        data = self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}, auth=True)
        for bal in data.get("result", {}).get("list", []):
            if bal.get("coin") == asset:
                return float(bal.get("walletBalance", 0))
        return 0.0

    def place_order(self, symbol: str, side: str, qty: float, order_type: str,
                    price: Optional[float] = None) -> Dict[str, any]:
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
        }
        if price is not None:
            params["price"] = str(price)
        return self._request("POST", "/v5/order/create", params, weight=1, auth=True)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        params = {"category": "linear", "symbol": symbol, "orderId": order_id}
        self._request("POST", "/v5/order/cancel", params, weight=1, auth=True)
        return True
