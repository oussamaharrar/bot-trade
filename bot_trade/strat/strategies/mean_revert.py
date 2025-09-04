from __future__ import annotations

import pandas as pd

from . import register, BaseStrategy


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
