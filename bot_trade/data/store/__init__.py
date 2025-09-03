from __future__ import annotations

"""Data store interfaces (parquet, calendar, registry)."""

from dataclasses import dataclass
from typing import Protocol, Sequence

class ParquetSource(Protocol):
    def read(self, symbol: str, frame: str) -> object: ...

@dataclass
class TradingCalendar:
    tz: str = "UTC"

    def is_open(self, ts) -> bool:  # pragma: no cover - placeholder
        return True

class SymbolRegistry(Protocol):
    def list_symbols(self) -> Sequence[str]: ...
    def frames_for(self, symbol: str) -> Sequence[str]: ...

__all__ = ["ParquetSource", "TradingCalendar", "SymbolRegistry"]
