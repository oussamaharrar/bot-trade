from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import numpy as np
import pandas as pd


def _to_series(df: pd.DataFrame, col: str) -> pd.Series:
    try:
        return pd.to_numeric(df[col], errors="coerce")
    except Exception:
        return pd.Series(dtype=float)


def detect_regime(df_like: Any, *, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Detect simple market regime from price/volume/spread features.

    Parameters
    ----------
    df_like : Any
        DataFrame-like object containing at least close, volume, bid, ask.
    cfg : dict
        Threshold configuration with windows and cut-offs.

    Returns
    -------
    dict
        {"name": str, "scores": dict, "ts": iso str}
    """
    try:
        df = pd.DataFrame(df_like).copy()
    except Exception:
        df = pd.DataFrame()
    ts = dt.datetime.utcnow().isoformat()
    if df.empty:
        return {"name": "unknown", "scores": {}, "ts": ts}

    try:
        idx = df.index[-1]
        if isinstance(idx, (pd.Timestamp, dt.datetime)):
            ts = idx.isoformat()
    except Exception:
        pass

    scores: Dict[str, Any] = {}
    thr = cfg.get("thresholds", cfg)

    close = _to_series(df, "close")
    returns = np.log(close).diff()
    try:
        vol_win = int(thr.get("volatility", {}).get("window", 20))
        vol = returns.rolling(vol_win).std().iloc[-1]
        scores["volatility"] = float(vol) if pd.notna(vol) else None
    except Exception:
        scores["volatility"] = None

    try:
        trend_win = int(thr.get("trend", {}).get("window", 20))
        log_price = np.log(close)
        s = log_price.tail(trend_win).dropna()
        if len(s) >= 2:
            x = np.arange(len(s))
            slope, _ = np.polyfit(x, s.values, 1)
        else:
            slope = float("nan")
        scores["trend_slope"] = float(slope) if not np.isnan(slope) else None
    except Exception:
        scores["trend_slope"] = None

    try:
        bid = _to_series(df, "bid")
        ask = _to_series(df, "ask")
        mid = (bid + ask) / 2.0
        sp_win = int(thr.get("spread_bp", {}).get("window", 1))
        spread_bp = ((ask - bid) / mid).rolling(sp_win).mean().iloc[-1] * 10000
        scores["spread_bp"] = float(spread_bp) if pd.notna(spread_bp) else None
    except Exception:
        scores["spread_bp"] = None

    try:
        vol_series = _to_series(df, "volume")
        volw = int(thr.get("volume", {}).get("window", 20))
        mean_vol = vol_series.rolling(volw).mean().iloc[-1]
        cur_vol = vol_series.iloc[-1]
        rel = cur_vol / mean_vol if mean_vol and not np.isnan(mean_vol) else float("nan")
        scores["volume_rel"] = float(rel) if not np.isnan(rel) else None
    except Exception:
        scores["volume_rel"] = None

    if any(v is None for v in scores.values()):
        return {"name": "unknown", "scores": scores, "ts": ts}

    regime = "range"
    vol_thr = thr.get("volatility", {}).get("high", float("inf"))
    trend_up_thr = thr.get("trend", {}).get("up", float("inf"))
    trend_dn_thr = thr.get("trend", {}).get("down", float("-inf"))
    spread_thr = thr.get("spread_bp", {}).get("high", float("inf"))
    vol_low_thr = thr.get("volume", {}).get("low", float("-inf"))

    if scores["volatility"] > vol_thr:
        regime = "high_vol"
    elif scores["volume_rel"] < vol_low_thr:
        regime = "low_liquidity"
    elif scores["spread_bp"] > spread_thr:
        regime = "low_liquidity"
    elif scores["trend_slope"] > trend_up_thr:
        regime = "trend_up"
    elif scores["trend_slope"] < trend_dn_thr:
        regime = "trend_down"
    else:
        regime = "range"

    return {"name": regime, "scores": scores, "ts": ts}

from pathlib import Path
from bot_trade.tools.atomic_io import append_jsonl


class RegimeDetector:
    """Incremental regime detector with optional JSONL logging."""

    def __init__(
        self,
        cfg: Dict[str, Any] | None = None,
        log_path: Path | None = None,
        seed: int | None = 0,
    ) -> None:
        self.cfg = cfg or {}
        self.log_path = Path(log_path) if log_path else None
        self.rng = np.random.default_rng(seed)
        self._wid = 0

    def update(self, df_slice: Any) -> Dict[str, Any]:
        info = detect_regime(df_slice, cfg=self.cfg)
        try:
            wid = int(len(df_slice))
        except Exception:
            wid = self._wid
            self._wid += 1
        info["window_id"] = wid
        self._wid = wid
    def __init__(self, cfg: Dict[str, Any] | None = None, log_path: Path | None = None) -> None:
        self.cfg = cfg or {}
        self.log_path = Path(log_path) if log_path else None

    def update(self, df_slice: Any) -> Dict[str, Any]:
        info = detect_regime(df_slice, cfg=self.cfg)
        if self.log_path:
            rec = {
                "ts": info.get("ts"),
                "regime": info.get("name"),
                "features": info.get("scores", {}),
                "window_id": info.get("window_id"),

            }
            try:
                append_jsonl(self.log_path, rec)
            except Exception:
                pass
        return info
