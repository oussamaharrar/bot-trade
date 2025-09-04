"""Live price feed with websocket and deterministic HTTP polling fallback."""

from __future__ import annotations

import json
import random
import threading
import time
from typing import Callable, Optional

import requests

from bot_trade.utils.rate_limit import RateLimiter

try:
    import websocket  # type: ignore
except Exception:  # pragma: no cover - optional
    websocket = None


class LiveFeed:
    def __init__(
        self, ws_url: Optional[str], http_url: str, interval: float = 1.0
    ) -> None:
        self.ws_url = ws_url
        self.http_url = http_url
        self.interval = interval
        # allow up to two polls per second; shared limiter ensures spacing
        self.limiter = RateLimiter(capacity=1, refill_time=0.5)

    # ------------------------------------------------------------------
    def stream(self, symbol: str, on_tick: Callable[[float], None]) -> None:
        target = self._ws_loop if websocket and self.ws_url else self._http_loop
        thread = threading.Thread(target=target, args=(symbol, on_tick), daemon=True)
        thread.start()

    def _ws_loop(
        self, symbol: str, on_tick: Callable[[float], None]
    ) -> None:  # pragma: no cover - network
        url = self.ws_url
        while True:
            try:
                ws = websocket.create_connection(url)  # type: ignore[attr-defined]
                if "binance" in url:
                    ws.send(
                        json.dumps(
                            {
                                "method": "SUBSCRIBE",
                                "params": [f"{symbol.lower()}@ticker"],
                                "id": 1,
                            }
                        )
                    )
                while True:
                    data = json.loads(ws.recv())
                    price = float(
                        data.get("c") or data.get("data", {}).get("p", 0)
                    )
                    on_tick(price)
            except Exception:
                # websocket unavailable -> fallback to HTTP polling
                self._http_loop(symbol, on_tick)
                return

    def _http_loop(self, symbol: str, on_tick: Callable[[float], None]) -> None:
        last = time.time()
        while True:
            self.limiter.acquire()
            price = 0.0
            try:
                r = requests.get(self.http_url, params={"symbol": symbol})
                data = r.json()
                price = float(
                    data.get("price")
                    or data.get("result", {}).get("list", [{}])[0].get("lastPrice", 0)
                )
                on_tick(price)
            except Exception:
                pass
            now = time.time()
            drift = int((now - last - self.interval) * 1000)
            print(f"[LIVE] poll price={price} drift_ms={drift}")
            last = now
            time.sleep(self.interval * (0.5 + random.random() * 0.5))
