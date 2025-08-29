import numpy as np
import pandas as pd

class RewardSignalTracker:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.w = (
            cfg.get('w_pnl', 1.0),
            cfg.get('w_volatility', 0.35),
            cfg.get('w_drawdown', 0.50),
            cfg.get('w_trend', 0.15),
            cfg.get('w_danger', 0.20),
            cfg.get('w_trade_cost', 0.1),
            cfg.get('w_dwell', 0.01),
        )
        self.freeze_zero = cfg.get('freeze_zero_reward', True)
        self.vol_w = cfg.get('vol_window', 32)
        self.trend_w = cfg.get('trend_window', 24)
        self.rew_hist = []

    def compute_trend_slope(self, series: pd.Series) -> float:
        s = series.dropna().tail(self.trend_w)
        if len(s) < 3:
            return 0.0
        x = np.arange(len(s))
        a, _ = np.polyfit(x, s.values.astype(float), 1)
        return float(a)

    def step_reward(self, pnl, close_series, atr=None,
                    drawdown=0.0, danger=False, freeze=False,
                    signal_trend_follow=False, ema_series=None,
                    trade_cost: float = 0.0, dwell_pen: float = 0.0):
        w_pnl, w_vol, w_dd, w_trend, w_danger, w_trade, w_dwell = self.w

        vol_pen = 0.0
        if close_series is not None:
            ret = pd.Series(close_series).pct_change()
            vol_pen = ret.rolling(self.vol_w).std().iloc[-1]
            if np.isnan(vol_pen):
                vol_pen = 0.0
        if atr is not None and vol_pen == 0.0:
            last_close = float(pd.Series(close_series).iloc[-1])
            vol_pen = float(atr) / max(last_close, 1e-9)

        base = pd.Series(ema_series) if ema_series is not None else pd.Series(close_series)
        trend_slope = self.compute_trend_slope(base)

        dd_pen = max(drawdown, 0.0)
        danger_pen = 1.0 if danger else 0.0

        reward = (
            w_pnl * float(pnl)
            - w_vol * float(vol_pen)
            - w_dd * float(dd_pen)
            - w_danger * float(danger_pen)
            - w_trade * float(trade_cost)
            - w_dwell * float(dwell_pen)
        )

        if signal_trend_follow and trend_slope > 0:
            reward += w_trend * float(trend_slope)

        if freeze and self.freeze_zero:
            reward = min(reward, 0.0)

        self.rew_hist.append(reward)
        return float(reward), {
            "vol_pen": float(vol_pen),
            "dd_pen": float(dd_pen),
            "trend_slope": float(trend_slope),
            "danger_pen": float(danger_pen),
            "trade_cost": float(trade_cost),
            "dwell_pen": float(dwell_pen),
        }
