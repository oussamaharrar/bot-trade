from __future__ import annotations
import math
import numpy as np
import pandas as pd


def z_score(s: pd.Series, eps: float = 1e-8) -> pd.Series | None:
    """Return z-score normalized series with clamping."""
    if s is None or len(s) == 0:
        return None
    mu = float(np.nanmean(s))
    sigma = float(np.nanstd(s))
    if not math.isfinite(mu) or not math.isfinite(sigma):
        return None
    sigma = sigma if sigma > eps else eps
    z = (s - mu) / sigma
    return z.clip(-8, 8)


def min_max(s: pd.Series) -> pd.Series | None:
    """Return min-max scaled series with NaN safeguards."""
    if s is None or len(s) == 0:
        return None
    lo = float(np.nanmin(s))
    hi = float(np.nanmax(s))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
        return None
    return (s - lo) / (hi - lo)
