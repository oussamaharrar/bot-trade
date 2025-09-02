from __future__ import annotations
import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from bot_trade.tools.atomic_io import append_jsonl

CONFIG: Dict[str, Any] = {}
STATE: Dict[str, int] = {"level": 0, "cool": 0}


def configure(cfg: Dict[str, Any]) -> None:
    """Set global configuration for strategy failure policy."""
    CONFIG.clear()
    CONFIG.update(cfg or {})


def _now() -> str:
    return dt.datetime.utcnow().isoformat()


def evaluate_step(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return zero or more failure events detected this step."""
    if not CONFIG.get("enabled", False):
        return []
    thr = CONFIG.get("thresholds", {}) or {}
    events: List[Dict[str, Any]] = []
    ts = ctx.get("ts") or _now()
    for key, limit in thr.items():
        val = ctx.get(key)
        if val is None or limit is None:
            continue
        try:
            v = float(val)
            lim = float(limit)
        except Exception:
            continue
        trig = v >= lim if key in {"loss_streak", "stuck_position_s", "partial_fill_timeout_s"} else v > lim
        if trig:
            events.append({
                "flag": key,
                "reason": f"{key} threshold",
                "value": v,
                "threshold": lim,
                "ts": ts,
            })
    return events


def _clamp(key: str, value: float) -> float:
    clamp_cfg = (CONFIG.get("clamps") or {}).get(key, {})
    lo = float(clamp_cfg.get("min", float("-inf")))
    hi = float(clamp_cfg.get("max", float("inf")))
    return max(lo, min(value, hi))


def apply_actions(
    events: List[Dict[str, Any]],
    *,
    risk_manager: Any = None,
    controller: Any = None,
    env: Any = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Escalate actions based on policy and record events."""
    summary: Dict[str, Any] = {"applied_actions": [], "new_risk_bounds": {}}
    if not events:
        if STATE.get("cool", 0) > 0:
            STATE["cool"] -= 1
            if STATE["cool"] <= 0:
                STATE["level"] = 0
        return summary

    STATE["cool"] = int(CONFIG.get("cool_down_steps", 0))
    actions = CONFIG.get("actions", [])
    level = STATE.get("level", 0)

    for ev in events:
        try:
            if risk_manager is not None:
                risk_manager.record_flag(ev["flag"], ev["reason"], ev["value"], ev["threshold"])
        except Exception:
            pass

    applied: List[str] = []
    new_bounds: Dict[str, float] = {}
    if level < len(actions):
        act = actions[level]
        if act == "reduce_risk" and risk_manager is not None:
            try:
                factor = 0.5
                if getattr(controller, "last_regime", None) in {"high_vol", "low_liquidity"}:
                    factor *= 0.8
                risk_manager.current_risk = _clamp("risk_scale", risk_manager.current_risk * factor)
                new_bounds["risk_scale"] = risk_manager.current_risk
                if env is not None and hasattr(env, "exec_sim"):
                    env.exec_sim.max_spread_bp = _clamp("max_spread_bp", getattr(env.exec_sim, "max_spread_bp", 0.0) * factor)
                    new_bounds["max_spread_bp"] = env.exec_sim.max_spread_bp
            except Exception:
                pass
        elif act == "freeze_trading" and env is not None:
            try:
                setattr(env, "trading_frozen", True)
            except Exception:
                pass
        elif act == "flat_all" and env is not None:
            try:
                flat = getattr(env, "flat_all", None)
                if callable(flat):
                    flat()
            except Exception:
                pass
        elif act == "halt_training":
            summary["halt"] = True
        applied.append(act)
        level += 1
        if controller is not None and getattr(controller, "log_path", None) and act != "warn":
            record = {"ts": _now(), "source": "safety", "action": act, "risk_bounds": new_bounds}
            try:
                append_jsonl(controller.log_path, record)
            except Exception:
                pass

    STATE["level"] = level

    if log_path is not None:
        for ev in events:
            rec = dict(ev)
            rec["actions"] = applied
            try:
                append_jsonl(log_path, rec)
            except Exception:
                pass

    summary["applied_actions"] = applied
    summary["new_risk_bounds"] = new_bounds
    return summary
