from __future__ import annotations

"""Simple strategy registry with a couple of built-ins."""

from typing import Callable, Dict, List

import pandas as pd


STRATEGY_REGISTRY: Dict[str, Callable[..., "BaseStrategy"]] = {}


def register(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        STRATEGY_REGISTRY[name] = cls  # type: ignore
        return cls

    return decorator


class BaseStrategy:
    def __init__(self, **params) -> None:
        self.params = params

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


@register("trend_following")
class TrendFollowingStrategy(BaseStrategy):
    def __call__(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover - math heavy
        fast = int(self.params.get("fast", 12))
        slow = int(self.params.get("slow", 26))
        ma_fast = df["close"].rolling(fast).mean()
        ma_slow = df["close"].rolling(slow).mean()
        return (ma_fast > ma_slow).astype(float) - (ma_fast < ma_slow).astype(float)


@register("mean_revert")
class MeanRevertStrategy(BaseStrategy):
    def __call__(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover - math heavy
        lb = int(self.params.get("lookback", 50))
        z = float(self.params.get("z", 1.0))
        pct = df["close"].pct_change()
        mean = pct.rolling(lb).mean()
        std = pct.rolling(lb).std()
        score = (pct - mean) / std
        return (score < -z).astype(float) - (score > z).astype(float)


def load_strategies(names: List[str], params: Dict[str, Dict]) -> List[BaseStrategy]:
    out: List[BaseStrategy] = []
    for name in names:
        cls = STRATEGY_REGISTRY.get(name)
        if not cls:
            continue
        out.append(cls(**params.get(name, {})))
    return out

