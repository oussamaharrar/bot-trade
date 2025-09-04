from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3.common.callbacks import BaseCallback

from bot_trade.tools.atomic_io import append_jsonl

from .risk_rules import RULES_MAP, RiskRule


class RiskRegistry:
    """Runtime registry holding risk rules and applying adjustments."""

    def __init__(self, rules: List[RiskRule], env: Any, log_path: Path | None = None) -> None:
        self.rules = rules
        self.env = env
        self.log_path = Path(log_path) if log_path else None

    @classmethod
    def from_yaml(cls, env: Any, path: Path | None, log_path: Path | None = None) -> "RiskRegistry":
        rules: List[RiskRule] = []
        if path and Path(path).exists():
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(Path(path).read_text()) or {}
                for item in cfg.get("rules", []):
                    name = item.get("rule")
                    params = item.get("params", {})
                    cls_ = RULES_MAP.get(name)
                    if cls_:
                        try:
                            rules.append(cls_(**params))
                        except Exception:
                            continue
            except Exception:
                pass
        return cls(rules, env, log_path)

    # --------------------------------------------------------------
    def evaluate(self, state: Dict[str, Any], step: int) -> None:
        for rule in self.rules:
            try:
                res = rule.evaluate(state)
            except Exception:
                continue
            if res.get("triggered"):
                self._apply_adjustments(res.get("adjustments", {}))
                rec = {
                    "ts": dt.datetime.utcnow().isoformat(),
                    "step": step,
                    "rule": rule.name,
                    **res,
                }
                if self.log_path:
                    try:
                        append_jsonl(self.log_path, rec)
                    except Exception:
                        pass

    # --------------------------------------------------------------
    def _apply_adjustments(self, adj: Dict[str, Any]) -> None:
        env = self.env
        re = getattr(env, "risk_engine", None)
        exec_sim = getattr(env, "exec_sim", None)
        if adj.get("freeze"):
            try:
                env.trading_frozen = True
            except Exception:
                pass
        if re:
            if "max_leverage" in adj:
                cur = getattr(re, "max_risk", None)
                val = float(adj["max_leverage"])
                if cur is None or val < cur:
                    re.max_risk = val
            if "trailing_dd_limit" in adj:
                cur = getattr(re, "max_drawdown_stop", None)
                val = float(adj["trailing_dd_limit"])
                if cur is None or val < cur:
                    re.max_drawdown_stop = val
            if "max_position" in adj:
                cur = getattr(re, "max_units", None)
                val = float(adj["max_position"])
                if cur is None or val < cur:
                    re.max_units = val
        if exec_sim and "max_spread_bp" in adj:
            cur = getattr(exec_sim, "max_spread_bp", None)
            val = float(adj["max_spread_bp"])
            if cur is None or val < cur:
                exec_sim.max_spread_bp = val


class RiskRegistryCallback(BaseCallback):
    def __init__(self, registry: RiskRegistry, every: int = 1) -> None:
        super().__init__()
        self.registry = registry
        self.every = max(1, int(every))

    def _on_step(self) -> bool:  # noqa: D401
        if self.n_calls % self.every != 0:
            return True
        infos = self.locals.get("infos")
        env = None
        try:
            env = self.training_env.envs[0]  # type: ignore[attr-defined]
        except Exception:
            try:
                env = self.training_env.venv.envs[0]  # type: ignore[attr-defined]
            except Exception:
                env = None
        info: Dict[str, Any] = {}
        if infos:
            try:
                info = infos[0]
            except Exception:
                info = {}
        state = {
            "reward": info.get("reward"),
            "pnl": info.get("pnl"),
            "drawdown": info.get("drawdown"),
            "loss_streak": info.get("loss_streak"),
            "spread_bp": getattr(env, "last_spread_bp", None) if env is not None else None,
            "depth": getattr(env, "last_depth", None) if env is not None else None,
            "gap_pct": getattr(env, "last_gap_pct", None) if env is not None else None,
            "regime": info.get("regime"),
            "step": int(self.num_timesteps),
        }
        self.registry.evaluate(state, int(self.num_timesteps))
        return True


__all__ = ["RiskRegistry", "RiskRegistryCallback"]
