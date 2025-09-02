from __future__ import annotations

import datetime as dt
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from .regime import detect_regime
from bot_trade.tools.atomic_io import append_jsonl


class AdaptiveController:
    """Adjust reward weights and risk bounds based on detected regime."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        env: Optional[Any] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg or {}
        self.env = env
        self.log_path = Path(log_path) if log_path else None
        self.warned = False
        self.last_regime = "unknown"
        self.dist: Counter[str] = Counter()

        # baselines
        rw = getattr(getattr(env, "reward_tracker", None), "w", None)
        self.base_weights = tuple(rw) if rw else None
        re = getattr(env, "risk_engine", None)
        ex = getattr(env, "exec_sim", None)
        self.base_bounds = {
            "max_spread_bp": getattr(ex, "max_spread_bp", None),
            "exposure_cap": getattr(re, "max_risk", None),
            "freeze_after_losses": getattr(re, "freeze_limit", None),
        }

    # ----------------------------------------------
    def update(self, df_slice: Any) -> None:
        info = detect_regime(df_slice, cfg=self.cfg)
        regime = info.get("name", "unknown")
        self.last_regime = regime
        self.dist[regime] += 1

        mapping = (self.cfg.get("mappings", {}) or {}).get(regime)
        if not mapping:
            if not self.warned and regime != "unknown":
                logging.warning("[REGIME] no mapping for %s", regime)
                self.warned = True
            weights = {}
            bounds = {}
        else:
            weights = mapping.get("reward_weights", {})
            bounds = mapping.get("risk_bounds", {})
        applied_w = self._apply_weights(weights)
        applied_b = self._apply_bounds(bounds)
        if self.log_path:
            record = {
                "ts": dt.datetime.utcnow().isoformat(),
                "regime": regime,
                "weights": applied_w,
                "risk_bounds": applied_b,
            }
            try:
                append_jsonl(self.log_path, record)
            except Exception:
                pass

    # ----------------------------------------------
    def _apply_weights(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        env = self.env
        tracker = getattr(env, "reward_tracker", None)
        if tracker is None or not weights:
            return {}
        w = list(tracker.w if hasattr(tracker, "w") else self.base_weights or [])
        if len(w) != 7:
            return {}
        changed: Dict[str, Any] = {}
        if "dd_penalty" in weights:
            w[2] = float(weights["dd_penalty"])
            changed["dd_penalty"] = w[2]
        if "trend_bonus" in weights:
            w[3] = float(weights["trend_bonus"])
            changed["trend_bonus"] = w[3]
        if "holding_penalty" in weights:
            w[6] = float(weights["holding_penalty"])
            changed["holding_penalty"] = w[6]
        tracker.w = tuple(w)
        return changed

    def _clamp(self, key: str, value: float) -> Optional[float]:
        lim = (self.cfg.get("bounds", {}) or {}).get(key, {})
        lo = lim.get("min", float("-inf"))
        hi = lim.get("max", float("inf"))
        if value < lo or value > hi:
            if not self.warned:
                logging.warning("[REGIME] %s=%s out of bounds [%s,%s]", key, value, lo, hi)
                self.warned = True
            return None
        return value

    def _apply_bounds(self, bounds: Dict[str, Any]) -> Dict[str, Any]:
        env = self.env
        re = getattr(env, "risk_engine", None)
        ex = getattr(env, "exec_sim", None)
        changed: Dict[str, Any] = {}
        for k, v in bounds.items():
            try:
                val = float(v)
            except Exception:
                continue
            val = self._clamp(k, val)
            if val is None:
                continue
            if k == "max_spread_bp" and ex is not None:
                ex.max_spread_bp = val
                changed[k] = val
            elif k == "exposure_cap" and re is not None:
                re.max_risk = val
                changed[k] = val
            elif k == "freeze_after_losses" and re is not None:
                re.freeze_limit = int(val)
                changed[k] = int(val)
        return changed

    # ----------------------------------------------
    def get_distribution(self) -> Dict[str, float]:
        total = sum(self.dist.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.dist.items()}
