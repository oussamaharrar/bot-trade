from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import yaml

# ---------------------------------------------------------------------------
# Reward term registry
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[[Any, Any, Mapping[str, Any], Mapping[str, Any]], float | None]] = {}


def _safe(val: Any) -> float | None:
    try:
        f = float(val)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def register(name: str):
    def _wrap(fn: Callable[[Any, Any, Mapping[str, Any], Mapping[str, Any]], float | None]):
        REGISTRY[name] = fn
        return fn
    return _wrap


@register("base_pnl")
def base_pnl(obs, action, info, state) -> float | None:
    return _safe(info.get("pnl"))


@register("risk_drawdown")
def risk_drawdown(obs, action, info, state) -> float | None:
    return _safe(info.get("drawdown"))


@register("inventory_penalty")
def inventory_penalty(obs, action, info, state) -> float | None:
    return _safe(state.get("inventory"))


@register("latency_penalty")
def latency_penalty(obs, action, info, state) -> float | None:
    return _safe(info.get("latency"))


@register("slippage_penalty")
def slippage_penalty(obs, action, info, state) -> float | None:
    return _safe(info.get("slippage"))


# ---------------------------------------------------------------------------
# Spec loading and reward computation
# ---------------------------------------------------------------------------


def load_reward_spec(src: str | Path | Mapping[str, Any] | None) -> Dict[str, Any]:
    if src is None:
        data: Dict[str, Any] = {}
    elif isinstance(src, (str, Path)):
        with Path(src).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = dict(src)
    weights = {k: float(v) for k, v in (data.get("weights") or {}).items()}
    clamp_data = data.get("clamps") or {}
    clamps: Dict[str, Dict[str, float]] = {}
    for k, v in clamp_data.items():
        clamps[k] = {
            "min": float(v.get("min", float("-inf"))),
            "max": float(v.get("max", float("inf"))),
        }
    gc = data.get("global_clamp") or {}
    global_clamp = {
        "min": float(gc.get("min", float("-inf"))),
        "max": float(gc.get("max", float("inf"))),
    }
    return {"weights": weights, "clamps": clamps, "global_clamp": global_clamp}


def evaluate_terms(obs, action, info: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, fn in REGISTRY.items():
        try:
            val = fn(obs, action, info, state)
        except Exception:
            val = None
        val = _safe(val)
        if val is not None:
            out[name] = float(val)
    return out


def compute_reward(
    terms: Mapping[str, float],
    weights: Mapping[str, float],
    clamps: Mapping[str, Mapping[str, float]],
    global_clamp: Mapping[str, float],
) -> float:
    total = 0.0
    for name, weight in weights.items():
        val = _safe(terms.get(name, 0.0))
        if val is None:
            val = 0.0
        if name in clamps:
            lo = clamps[name].get("min", float("-inf"))
            hi = clamps[name].get("max", float("inf"))
            val = max(lo, min(val, hi))
        total += weight * val
    lo = global_clamp.get("min", float("-inf"))
    hi = global_clamp.get("max", float("inf"))
    total = max(lo, min(total, hi))
    total = _safe(total) or 0.0
    return float(total)
