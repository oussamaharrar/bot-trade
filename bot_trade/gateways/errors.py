from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GatewayError(RuntimeError):
    """Structured gateway error with code and context."""

    code: int
    context: Any

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        super().__init__(str(self.context))


__all__ = ["GatewayError"]
