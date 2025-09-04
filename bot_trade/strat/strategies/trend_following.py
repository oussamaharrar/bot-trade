from __future__ import annotations

import pandas as pd

from . import register, BaseStrategy


@register("trend_following")
class TrendFollowingStrategy(BaseStrategy):
    def __call__(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover - math heavy
        fast = int(self.params.get("fast", 12))
        slow = int(self.params.get("slow", 26))
        ma_fast = df["close"].rolling(fast).mean()
        ma_slow = df["close"].rolling(slow).mean()
        return (ma_fast > ma_slow).astype(float) - (ma_fast < ma_slow).astype(float)
