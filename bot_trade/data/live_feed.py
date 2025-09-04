from __future__ import annotations
"""Live price feed with websocket + HTTP fallback."""

import json
import threading
import time
from typing import Callable, Optional

import requests

try:
    import websocket  # type: ignore
except Exception:  # pragma: no cover - optional
    websocket = None


class LiveFeed:
    def __init__(self, ws_url: Optional[str], http_url: str, interval: float = 1.0) -> None:
        self.ws_url = ws_url
        self.http_url = http_url
        self.interval = interval

    # ------------------------------------------------------------------
    def stream(self, symbol: str, on_tick: Callable[[float], None]) -> None:
        if websocket and self.ws_url:
            thread = threading.Thread(
                target=self._ws_loop, args=(symbol, on_tick), daemon=True
            )
            thread.start()
        else:
            thread = threading.Thread(
                target=self._http_loop, args=(symbol, on_tick), daemon=True
            )
            thread.start()

    def _ws_loop(self, symbol: str, on_tick: Callable[[float], None]) -> None:  # pragma: no cover - network
        url = self.ws_url
        while True:
            try:
                ws = websocket.create_connection(url)  # type: ignore[attr-defined]
                if "binance" in url:
                    ws.send(json.dumps({"method": "SUBSCRIBE", "params": [f"{symbol.lower()}@ticker"], "id": 1}))
                while True:
                    data = json.loads(ws.recv())
                    price = float(data.get("c") or data.get("data", {}).get("p", 0))
                    on_tick(price)
            except Exception:
                time.sleep(self.interval)

    def _http_loop(self, symbol: str, on_tick: Callable[[float], None]) -> None:
        while True:
            try:
                r = requests.get(self.http_url, params={"symbol": symbol})
                data = r.json()
                price = float(data.get("price") or data.get("result", {}).get("list", [{}])[0].get("lastPrice", 0))
                on_tick(price)
            except Exception:
                pass
            time.sleep(self.interval)
