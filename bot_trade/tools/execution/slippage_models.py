"""Registry of slippage model functions."""
from __future__ import annotations
from typing import Callable, Dict, Optional

SlippageFn = Callable[[Dict[str, float], Optional[float], Optional[float]], float]


def _fixed(params: Dict[str, float], vol: Optional[float], depth: Optional[float]) -> float:
    return float(params.get("bp", 0.0))


def _vol_aware(params: Dict[str, float], vol: Optional[float], depth: Optional[float]) -> float:
    k = float(params.get("k", 0.0))
    return k * float(vol or 0.0)


def _depth_aware(params: Dict[str, float], vol: Optional[float], depth: Optional[float]) -> float:
    impact = float(params.get("book_impact", 0.0))
    d = float(depth or 0.0)
    if d <= 0:
        return 0.0
    return impact / d

_REGISTRY: Dict[str, SlippageFn] = {
    "fixed": _fixed,
    "fixed_bp": _fixed,
    "vol_aware": _vol_aware,
    "depth_aware": _depth_aware,
}


def get_slippage_model(name: str) -> SlippageFn:
    """Return slippage function for ``name`` or a no-op."""
    return _REGISTRY.get(name.lower(), _fixed)

__all__ = ["get_slippage_model", "SlippageFn"]
