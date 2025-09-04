from __future__ import annotations

"""Abstract market data collector interface.

Collectors are simple classes exposing a ``load`` method returning a pandas
``DataFrame`` with timezone aware UTC index and OHLCV + book fields.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class MarketCollector(ABC):
    """Base class for market data collectors."""

    @abstractmethod
    def load(
        self,
        symbol: str,
        frame: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a dataframe indexed by UTC timestamps.

        Implementations must return at least the columns ``open``, ``high``,
        ``low``, ``close`` and ``volume``. Additional optional columns are
        ``spread_bp``, ``best_bid``, ``best_ask`` and ``depth_top``.
        """
        raise NotImplementedError
