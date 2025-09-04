from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

STRATEGY_REGISTRY: Dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        STRATEGY_REGISTRY[name] = cls
        return cls

    return deco


class BaseStrategy:
    def __init__(self, **params) -> None:
        self.params = params

    def __call__(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError


def load_strategies(names: List[str], params: Dict[str, Dict]) -> List[BaseStrategy]:
    out: List[BaseStrategy] = []
    for name in names:
        cls = STRATEGY_REGISTRY.get(name)
        if not cls:
            continue
        out.append(cls(**params.get(name, {})))
    return out
