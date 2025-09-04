from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from bot_trade.tools.atomic_io import append_jsonl

from .regime import RegimeDetector


class AdaptiveController:
    """Apply reward and risk adjustments based on detected regimes."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        env: Optional[Any] = None,
        log_path: Optional[Path] = None,
        regime_log_path: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg or {}
        self.env = env
        self.log_path = Path(log_path) if log_path else None
        self.detector = RegimeDetector(self.cfg.get("detector", {}), regime_log_path)
        self.last_regime = "unknown"
        self.dist: Counter[str] = Counter()
        if self.log_path:
            try:
                self.log_path.touch()
            except Exception:
                self.log_path = None

        self.w_clamp = self.cfg.get("clamp", {}).get("weight_delta", {})
        self.r_clamp = self.cfg.get("clamp", {}).get("risk_delta", {})

        rw = getattr(getattr(env, "reward_tracker", None), "w", None)
        self.base_weights = list(rw) if rw else []
        re = getattr(env, "risk_engine", None)
        exec_sim = getattr(env, "exec_sim", None)
        self.base_bounds = {
            "max_position": getattr(re, "max_units", None),
            "max_leverage": getattr(re, "max_risk", None),
            "trailing_dd_limit": getattr(re, "max_drawdown_stop", None),
            "max_spread_bp": getattr(exec_sim, "max_spread_bp", None),
        }

    # --------------------------------------------------------------
    def update(self, df_slice: Any) -> None:
        info = self.detector.update(df_slice)
        regime = info.get("name", "unknown")
        self.last_regime = regime
        self.dist[regime] += 1
        w_applied = self._apply_weight_delta(self.cfg.get("weights_delta", {}).get(regime, {}))
        r_applied = self._apply_risk_delta(self.cfg.get("risk_delta", {}).get(regime, {}))
        if self.log_path and (w_applied or r_applied):
            rec = {
                "ts": dt.datetime.utcnow().isoformat(),
                "regime": regime,
                "weights": w_applied,
                "risk": r_applied,
            }
            try:
                append_jsonl(self.log_path, rec)
            except Exception:
                pass
        if self.env is not None:
            try:
                self.env.current_regime = regime
            except Exception:
                pass

    # --------------------------------------------------------------
    def _clamp(self, value: float, clamp: Dict[str, float]) -> float:
        lo = float(clamp.get("min", float("-inf")))
        hi = float(clamp.get("max", float("inf")))
        return max(lo, min(value, hi))

    def _apply_weight_delta(self, delta: Dict[str, float]) -> Dict[str, float]:
        tracker = getattr(self.env, "reward_tracker", None)
        if tracker is None or not self.base_weights:
            return {}
        mapping = {
            "base_pnl": 0,
            "inventory_penalty": 6,
            "risk_drawdown": 2,
            "slippage_penalty": 5,
        }
        applied: Dict[str, float] = {}
        w = list(self.base_weights)
        for k, v in (delta or {}).items():
            idx = mapping.get(k)
            if idx is None or idx >= len(w):
                continue
            adj = self._clamp(float(v), self.w_clamp)
            if adj == 0:
                continue
            w[idx] = float(w[idx]) + adj
            applied[k] = adj
        if applied:
            tracker.w = tuple(w)
            self.base_weights = w
        return applied

    def _apply_risk_delta(self, delta: Dict[str, float]) -> Dict[str, float]:
        re = getattr(self.env, "risk_engine", None)
        exec_sim = getattr(self.env, "exec_sim", None)
        applied: Dict[str, float] = {}
        for k, v in (delta or {}).items():
            if k not in self.base_bounds:
                continue
            base = self.base_bounds.get(k)
            if base is None:
                continue
            adj = self._clamp(float(v), self.r_clamp)
            new_val = base * (1.0 + adj)
            if k == "max_position" and re is not None:
                re.max_units = new_val
            elif k == "max_leverage" and re is not None:
                re.max_risk = new_val
            elif k == "trailing_dd_limit" and re is not None:
                re.max_drawdown_stop = new_val
            elif k == "max_spread_bp" and exec_sim is not None:
                exec_sim.max_spread_bp = new_val
            else:
                continue
            self.base_bounds[k] = new_val
            applied[k] = adj
        return applied

    # --------------------------------------------------------------
    def get_distribution(self) -> Dict[str, float]:
        total = sum(self.dist.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.dist.items()}


__all__ = ["AdaptiveController"]
