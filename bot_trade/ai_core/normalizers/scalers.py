from __future__ import annotations
import math
import numpy as np
import pandas as pd


def z_score(s: pd.Series) -> pd.Series | None:
    """Return z-score normalized series with NaN safeguards."""
    if s is None or len(s) == 0:
        return None
    mu = float(np.nanmean(s))
    sigma = float(np.nanstd(s))
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma == 0:
        return None
    return (s - mu) / sigma


def min_max(s: pd.Series) -> pd.Series | None:
    """Return min-max scaled series with NaN safeguards."""
    if s is None or len(s) == 0:
        return None
    lo = float(np.nanmin(s))
    hi = float(np.nanmax(s))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
        return None
    return (s - lo) / (hi - lo)
