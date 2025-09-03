from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from .regime import RegimeDetector
from bot_trade.tools.atomic_io import append_jsonl


class AdaptiveController:
    """Adjust reward weights and risk bounds based on detected regime."""

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
        det_cfg = self.cfg.get("detector", {})
        self.detector = RegimeDetector(det_cfg, regime_log_path, seed=det_cfg.get("seed"))
        self.detector = RegimeDetector(self.cfg.get("detector", {}), regime_log_path)
        self.last_regime = "unknown"
        self.dist: Counter[str] = Counter()
        self._last_key: tuple[str, int] | None = None

        self.w_clamp = self.cfg.get("clamps", {}).get("reward_weight_delta", {})
        self.b_clamp = self.cfg.get("clamps", {}).get("risk_bound_delta", {})

        self.w_clamp = self.cfg.get("clamps", {}).get("reward_weight_delta", {})
        self.b_clamp = self.cfg.get("clamps", {}).get("risk_bound_delta", {})

        rw = getattr(getattr(env, "reward_tracker", None), "w", None)
        self.base_weights = list(rw) if rw else []
        re = getattr(env, "risk_engine", None)
        self.base_bounds = {
            "max_position": getattr(re, "max_units", None),
            "max_leverage": getattr(re, "max_risk", None),
            "trailing_dd_limit": getattr(re, "max_drawdown_stop", None),
        }

    # --------------------------------------------------------------
    def update(self, df_slice: Any) -> None:
        info = self.detector.update(df_slice)
        regime = info.get("name", "unknown")
        wid = int(info.get("window_id", 0))
        self.last_regime = regime
        self.dist[regime] += 1
        key = (regime, wid)
        if regime != getattr(self, "_last_print_regime", None) and key != self._last_key:
            mapping = (self.cfg.get("regime_rules") or {}).get(regime, {})
            d_w = self._apply_reward_delta(mapping.get("reward_delta", {}))
            d_r = self._apply_risk_delta(mapping.get("risk_clamp_delta", {}))
            print(f"[ADAPT] regime={regime} dW={list(d_w.keys())} dRisk={list(d_r.keys())}")
            if self.log_path:
                rec = {
                    "ts": dt.datetime.utcnow().isoformat(),
                    "regime": regime,
                    "window_id": wid,
                    "dW": d_w,
                    "dRisk": d_r,
                }
                try:
                    append_jsonl(self.log_path, rec)
                except Exception:
                    pass
            self._last_key = key
            self._last_print_regime = regime

    def _clamp(self, value: float, clamp: Dict[str, float]) -> float:
        lo = float(clamp.get("min", float("-inf")))
        hi = float(clamp.get("max", float("inf")))
        return max(lo, min(value, hi))


        mapping = (self.cfg.get("regime_rules") or {}).get(regime, {})
        d_w = self._apply_reward_delta(mapping.get("reward_delta", {}))
        d_r = self._apply_risk_delta(mapping.get("risk_clamp_delta", {}))
        print(f"[ADAPT] regime={regime} dW={list(d_w.keys())} dRisk={list(d_r.keys())}")
        if self.log_path:
            rec = {
                "ts": dt.datetime.utcnow().isoformat(),
                "regime": regime,
                "dW": d_w,
                "dRisk": d_r,
            }
            try:
                append_jsonl(self.log_path, rec)
            except Exception:
                pass

    # --------------------------------------------------------------
    def _clamp(self, value: float, clamp: Dict[str, float]) -> float:
        lo = float(clamp.get("min", float("-inf")))
        hi = float(clamp.get("max", float("inf")))
        return max(lo, min(value, hi))

    def _apply_reward_delta(self, delta: Dict[str, float]) -> Dict[str, float]:
        tracker = getattr(self.env, "reward_tracker", None)
        if tracker is None or not self.base_weights:
            return {}
        w = list(self.base_weights)
        mapping = {
            "base_pnl": 0,
            "risk_drawdown": 2,
            "slippage_penalty": 5,
            "inventory_penalty": 6,
        }
        applied: Dict[str, float] = {}
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
        if re is None:
            return {}
        applied: Dict[str, float] = {}
        for k, v in (delta or {}).items():
            if k not in self.base_bounds:
                continue
            base = self.base_bounds.get(k)
            if base is None:
                continue
            adj = self._clamp(float(v), self.b_clamp)
            if adj == 0:
                continue
            new_val = base + adj
            if k == "max_position":
                re.max_units = new_val
            elif k == "max_leverage":
                re.max_risk = new_val
            elif k == "trailing_dd_limit":
                re.max_drawdown_stop = new_val
            applied[k] = adj
            self.base_bounds[k] = new_val
        return applied

    # --------------------------------------------------------------
    def get_distribution(self) -> Dict[str, float]:
        total = sum(self.dist.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.dist.items()}
