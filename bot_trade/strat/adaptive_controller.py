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
        if self.env is not None:
            try:
                setattr(self.env, "current_regime", regime)
            except Exception:
                pass

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
        if tracker is None:
            return {}
        w = list(self.base_weights or getattr(tracker, "w", []))
        if len(w) != 7:
            return {}
        applied = {
            "dd_penalty": w[2],
            "trend_bonus": w[3],
            "holding_penalty": w[6],
        }
        if weights:
            if "dd_penalty" in weights:
                try:
                    val = self._clamp("dd_penalty", float(weights["dd_penalty"]), kind="weights")
                    w[2] = val
                    applied["dd_penalty"] = val
                except Exception:
                    if not self.warned:
                        logging.warning("[REGIME] invalid dd_penalty")
                        self.warned = True
            if "trend_bonus" in weights:
                try:
                    val = self._clamp("trend_bonus", float(weights["trend_bonus"]), kind="weights")
                    w[3] = val
                    applied["trend_bonus"] = val
                except Exception:
                    if not self.warned:
                        logging.warning("[REGIME] invalid trend_bonus")
                        self.warned = True
            if "holding_penalty" in weights:
                try:
                    val = self._clamp("holding_penalty", float(weights["holding_penalty"]), kind="weights")
                    w[6] = val
                    applied["holding_penalty"] = val
                except Exception:
                    if not self.warned:
                        logging.warning("[REGIME] invalid holding_penalty")
                        self.warned = True
        tracker.w = tuple(w)
        return applied

    def _clamp(self, key: str, value: float, kind: str = "bounds") -> Optional[float]:
        if kind == "bounds":
            cfg = self.cfg.get("bounds", {}) or {}
        else:
            cfg = self.cfg.get("weight_limits", {}) or {}
        lim = cfg.get(key, {})
        lo = float(lim.get("min", float("-inf")))
        hi = float(lim.get("max", float("inf")))
        clamped = max(lo, min(value, hi))
        if clamped != value and not self.warned:
            logging.warning("[REGIME] %s=%s clamped to [%s,%s]", key, value, lo, hi)
            self.warned = True
        return clamped

    def _apply_bounds(self, bounds: Dict[str, Any]) -> Dict[str, Any]:
        env = self.env
        re = getattr(env, "risk_engine", None)
        ex = getattr(env, "exec_sim", None)
        base = dict(self.base_bounds)
        if bounds:
            base.update(bounds)
        applied: Dict[str, Any] = {}
        if "max_spread_bp" in base and ex is not None:
            try:
                val = self._clamp("max_spread_bp", float(base["max_spread_bp"]))
                ex.max_spread_bp = val
                applied["max_spread_bp"] = val
            except Exception:
                pass
        if "exposure_cap" in base and re is not None:
            try:
                val = self._clamp("exposure_cap", float(base["exposure_cap"]))
                re.max_risk = val
                applied["exposure_cap"] = val
            except Exception:
                pass
        if "freeze_after_losses" in base and re is not None:
            try:
                val = self._clamp("freeze_after_losses", float(base["freeze_after_losses"]))
                re.freeze_limit = int(val)
                applied["freeze_after_losses"] = int(val)
            except Exception:
                pass
        return applied

    # ----------------------------------------------
    def get_distribution(self) -> Dict[str, float]:
        total = sum(self.dist.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.dist.items()}
