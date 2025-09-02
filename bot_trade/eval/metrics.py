from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_PERIODS_PER_YEAR = {
    "daily": 252,
    "hourly": 24 * 252,
    "minute": 60 * 24 * 252,
}


def _coerce_series(data):
    import pandas as pd

    if data is None:
        return pd.Series(dtype=float)
    if isinstance(data, pd.Series):
        s = data.copy()
    else:
        s = pd.Series(list(data), dtype=float)
    s = pd.to_numeric(s, errors="coerce")
    return s.dropna()


def to_equity_from_returns(returns, start: float = 1.0):
    import pandas as pd

    s = _coerce_series(returns)
    if s.empty:
        return pd.Series([start], dtype=float)
    return s.cumsum().add(start)


def sharpe(returns: Iterable[float] | 'pd.Series', rf: float = 0.0, period: str = "daily") -> Optional[float]:
    import numpy as np

    r = _coerce_series(returns)
    if r.empty:
        return None
    excess = r - rf
    std = excess.std()
    if std <= 1e-12 or np.isnan(std):
        return None
    mean = excess.mean()
    if np.isnan(mean):
        return None
    ratio = mean / std
    if not np.isfinite(ratio):
        return None
    scale = np.sqrt(_PERIODS_PER_YEAR.get(period, 252))
    return float(ratio * scale)


def sortino(returns: Iterable[float] | 'pd.Series', rf: float = 0.0, period: str = "daily") -> Optional[float]:
    import numpy as np

    r = _coerce_series(returns)
    if r.empty:
        return None
    excess = r - rf
    downside = excess[excess < 0]
    std = downside.std()
    if std <= 1e-12 or np.isnan(std):
        return None
    mean = excess.mean()
    if np.isnan(mean):
        return None
    ratio = mean / std
    if not np.isfinite(ratio):
        return None
    scale = np.sqrt(_PERIODS_PER_YEAR.get(period, 252))
    return float(ratio * scale)


def max_drawdown(equity_curve: Iterable[float] | 'pd.Series') -> float:
    import numpy as np

    e = _coerce_series(equity_curve)
    if e.empty:
        return 0.0
    roll_max = e.cummax()
    denom = roll_max.replace(0, np.nan)
    drawdown = e / denom - 1.0
    drawdown = drawdown.dropna()
    return float(abs(drawdown.min())) if not drawdown.empty else 0.0


def calmar(equity_curve: Iterable[float] | 'pd.Series') -> Optional[float]:
    import numpy as np

    e = _coerce_series(equity_curve)
    if e.empty:
        return None
    dd = max_drawdown(e)
    if dd <= 1e-12 or not np.isfinite(dd):
        return None
    start = e.iloc[0]
    total_return = (e.iloc[-1] - start) / abs(start) if start != 0 else 0.0
    ratio = total_return / dd if dd > 0 else None
    if ratio is None or not np.isfinite(ratio):
        return None
    return float(ratio)


def turnover(trades_df: 'pd.DataFrame') -> float:
    import pandas as pd
    import numpy as np

    if trades_df is None or trades_df.empty or "position" not in trades_df:
        return 0.0
    pos = pd.to_numeric(trades_df["position"], errors="coerce").dropna()
    if pos.empty:
        return 0.0
    delta = pos.diff().abs().sum()
    avg_exposure = pos.abs().mean()
    if avg_exposure == 0 or np.isnan(avg_exposure):
        return 0.0
    return float(delta / avg_exposure)


def slippage_proxy(trades_df: 'pd.DataFrame') -> Optional[float]:
    import pandas as pd
    import numpy as np

    if trades_df is None or trades_df.empty:
        return None
    col_fill = "fill_price"
    col_mid = "mid_price" if "mid_price" in trades_df.columns else "mid"
    if col_fill not in trades_df or col_mid not in trades_df:
        return None
    fill = pd.to_numeric(trades_df[col_fill], errors="coerce")
    mid = pd.to_numeric(trades_df[col_mid], errors="coerce")
    df = pd.DataFrame({"fill": fill, "mid": mid}).dropna()
    if df.empty:
        return None
    proxy = (df["fill"] - df["mid"]).abs() / df["mid"].replace(0, np.nan)
    proxy = proxy.dropna()
    if proxy.empty:
        return None
    return float(proxy.mean())


def win_rate(trades_df: 'pd.DataFrame') -> Optional[float]:
    import pandas as pd

    if trades_df is None or trades_df.empty or "pnl" not in trades_df:
        return None
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return None
    return float((pnl > 0).mean())


def avg_trade_pnl(trades_df: 'pd.DataFrame') -> Optional[float]:
    import pandas as pd

    if trades_df is None or trades_df.empty or "pnl" not in trades_df:
        return None
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return None
    return float(pnl.mean())
