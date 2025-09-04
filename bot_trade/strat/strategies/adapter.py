from __future__ import annotations


def to_discrete(a: float, thr: float = 0.1) -> int:
    """Map continuous action ``a`` to {-1,0,1} with tolerance ``thr``."""

    return 1 if a > thr else (-1 if a < -thr else 0)
