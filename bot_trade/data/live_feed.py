"""Live price feed with websocket and deterministic HTTP polling fallback."""

from __future__ import annotations

import json
import math
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
        self,
        ws_url: Optional[str],
        http_url: str,
        interval: float = 1.0,
        *,
        alpha: float = 0.3,
        bootstrap_price: Optional[float] = None,
    ) -> None:
        self.ws_url = ws_url
        self.http_url = http_url
        self.interval = interval
        # allow up to two polls per second; shared limiter ensures spacing
        self.limiter = RateLimiter(capacity=1, refill_time=0.5)
        self.alpha = alpha
        self.last_price: Optional[float] = bootstrap_price
        self.smooth_price: Optional[float] = bootstrap_price

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
        bad_seq = 0
        backoff_ms = 250
        while True:
            self.limiter.acquire()
            price: Optional[float] = None
            try:
                r = requests.get(self.http_url, params={"symbol": symbol})
                data = r.json()
                price = float(
                    data.get("price")
                    or data.get("result", {}).get("list", [{}])[0].get("lastPrice", 0)
                )
            except Exception:
                price = None

            if price is None or not math.isfinite(price) or price <= 0:
                bad_seq += 1
                if self.last_price is None:
                    sleep = min(backoff_ms, 2000) / 1000.0
                    print(f"[LIVE] bad_price; skipping; backoff_ms={int(sleep * 1000)}")
                    backoff_ms = int(min(backoff_ms * 1.6, 2000))
                    time.sleep(sleep)
                    continue
                price = self.last_price
            else:
                self.last_price = price
                bad_seq = 0
                backoff_ms = 250

            if self.smooth_price is None:
                self.smooth_price = price
            else:
                self.smooth_price = (
                    self.alpha * price + (1 - self.alpha) * self.smooth_price
                )
            ps = self.smooth_price
            on_tick(ps)

            now = time.time()
            drift = int((now - last - self.interval) * 1000)
            sleep = self.interval * (0.5 + random.random() * 0.5)
            if bad_seq:
                sleep = max(sleep, backoff_ms / 1000.0)
                backoff_ms = int(min(backoff_ms * 1.6, 2000))
            print(
                f"[LIVE] poll price={price:.2f} smooth={ps:.2f} drift_ms={drift} "
                f"bad_seq={bad_seq} sleep_ms={int(sleep * 1000)}"
            )
            last = now
            time.sleep(sleep)
