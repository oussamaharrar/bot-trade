from __future__ import annotations

"""Websocket collector using ccxt.pro when available.

If ccxt.pro is missing the collector falls back to :class:`CCXTRestCollector`
with a log line.
"""

from dataclasses import dataclass

import pandas as pd

from .base import CollectorConfig, MarketCollector
from .ccxt_rest_collector import CCXTRestCollector


@dataclass
class CCXTWSCollector(MarketCollector):
    exchange: str

    def load(self, cfg: CollectorConfig) -> pd.DataFrame:  # pragma: no cover - thin wrapper
        try:
            import ccxtpro  # type: ignore
        except Exception:
            print("[DATA] ccxt.pro not available; falling back to REST polling")
            rest = CCXTRestCollector(self.exchange)
            return rest.load(cfg)
        # Simple fallback to REST as full websocket support is out of scope
        rest = CCXTRestCollector(self.exchange)
        return rest.load(cfg)
