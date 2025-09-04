from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from bot_trade.tools.atomic_io import append_jsonl


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    try:
        return pd.to_numeric(df[col], errors="coerce")
    except Exception:
        return pd.Series(dtype=float)


def detect_regime(df_like: Any, *, cfg: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame(df_like).copy()
    ts = dt.datetime.utcnow().isoformat()
    if df.empty:
        return {"name": "unknown", "scores": {}, "ts": ts}

    scores: Dict[str, Any] = {}
    close = _series(df, "close")
    returns = close.pct_change()
    scores["vol_pct"] = float(returns.std() * 100) if len(returns.dropna()) else None

    atr = _series(df, "atr")
    scores["atr"] = float(atr.iloc[-1]) if not atr.empty else None

    adx = _series(df, "adx")
    scores["adx"] = float(adx.iloc[-1]) if not adx.empty else None

    rsi = _series(df, "rsi_14")
    scores["rsi"] = float(rsi.iloc[-1]) if not rsi.empty else None

    spread = _series(df, "spread")
    mid = close if not close.empty else pd.Series([1.0])
    if not spread.empty and not mid.empty:
        sp_bp = (spread.iloc[-1] / mid.iloc[-1]) * 10_000
    else:
        sp_bp = float("nan")
    scores["spread_bp"] = float(sp_bp) if not np.isnan(sp_bp) else None

    depth = _series(df, "depth")
    scores["depth"] = float(depth.iloc[-1]) if not depth.empty else None

    gap = returns.iloc[-1] if len(returns) else 0.0
    scores["gap_pct"] = float(gap)

    thr = cfg.get("thresholds", {})
    regime = "range"
    try:
        if scores["spread_bp"] is not None and scores["depth"] is not None:
            if scores["spread_bp"] > float(thr.get("illiquid", {}).get("spread_bp", 1e9)) or scores["depth"] < float(
                thr.get("illiquid", {}).get("depth", -1e9)
            ):
                regime = "illiquid"
        if regime == "range" and abs(scores["gap_pct"]) > float(thr.get("gap_risk", {}).get("gap_pct", 1e9)):
            regime = "gap_risk"
        if regime == "range" and scores["vol_pct"] is not None:
            if scores["vol_pct"] > float(thr.get("high_vol", {}).get("vol", 1e9)):
                regime = "high_vol"
            elif scores["vol_pct"] < float(thr.get("low_vol", {}).get("vol", -1e9)):
                regime = "low_vol"
        if regime == "range":
            slope = returns.tail(int(cfg.get("trend_window", 20))).mean()
            if scores["adx"] is not None and scores["adx"] > float(thr.get("trend", {}).get("adx", 25)):
                if slope > 0:
                    regime = "bull_trend"
                elif slope < 0:
                    regime = "bear_trend"
    except Exception:
        regime = "unknown"

    return {"name": regime, "scores": scores, "ts": ts}


class RegimeDetector:
    """Incremental regime detector with optional logging."""

    def __init__(self, cfg: Dict[str, Any] | None = None, log_path: Path | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or {}
        self.log_path = Path(log_path) if log_path else None
        self.current: Dict[str, Any] = {"name": "unknown", "scores": {}, "ts": dt.datetime.utcnow().isoformat()}

    @property
    def current_regime(self) -> str:
        return self.current.get("name", "unknown")

    def update(self, df_slice: Any) -> Dict[str, Any]:
        info = detect_regime(df_slice, cfg=self.cfg)
        self.current = info
        if self.log_path:
            rec = {"ts": info.get("ts"), "regime": info.get("name"), "scores": info.get("scores", {})}
            try:
                append_jsonl(self.log_path, rec)
            except Exception:
                pass
        return info


__all__ = ["RegimeDetector", "detect_regime"]
