"""
config/signals_bridge.py — Adapter layer to standardize signals across the system

Why:
- Keep original modules in config/signals/* intact.
- Provide a unified, structured API for env_trading/RiskManager/Callbacks.
- Central place for mapping human names -> standard machine keys and config thresholds.

Key exports:
- map_human_to_standard(human_names: list[str]) -> dict[str,int]
- entry_from_features(df) -> dict[str,int]     # vectorized, prefers columns from strategy_features
- entry_from_module(human_list) -> dict[str,int]  # uses entry_signals.py outputs
- danger_from_module(ctx) -> dict[str,int]
- freeze_from_module(ctx) -> dict[str,int]
- recovery_from_module(ctx) -> dict[str,int]
- reward_blend(base_reward: float, signals: dict[str,float], weights: dict[str,float]) -> float

All functions are side-effect free and do not mutate inputs.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any

# Optional YAML config for thresholds/mapping
try:
    import yaml
    CFG_PATH = os.path.join("config", "config.yaml")
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        _CFG = yaml.safe_load(f) or {}
except Exception:
    _CFG = {}

SIGCFG: Dict[str, Any] = _CFG.get("signals", {})
MAPCFG: Dict[str, Any] = SIGCFG.get("mapping", {})

# -------------------------------------------------------------------
# Mapping: human -> standard machine keys (override via config.yaml)
# -------------------------------------------------------------------
_DEFAULT_MAP = {
    # Entry (bullish)
    "RSI recovery": "signal_mean_reversion",
    "MACD bullish": "signal_trend_follow",
    "EMA crossover": "signal_trend_follow",
    "Bollinger squeeze": "signal_breakout_buy",
    "breakout_buy": "signal_breakout_buy",
    # Entry (bearish / risk)
    "breakout_sell": "signal_breakout_sell",
    # Aliases
    "rsi_recovery": "signal_mean_reversion",
    "macd_bullish": "signal_trend_follow",
}

HUMAN2STD: Dict[str, str] = {**_DEFAULT_MAP, **MAPCFG}


def map_human_to_standard(human_names: List[str]) -> Dict[str, int]:
    """Return a dict of standardized flags from a list of human-readable names.
    Unknown names are ignored. Multiple names can map to the same std key.
    """
    out: Dict[str, int] = {}
    for name in human_names or []:
        key = HUMAN2STD.get(str(name).strip(), None)
        if not key:
            continue
        out[key] = 1
    return out


# -------------------------------------------------------------------
# Entry signals from features (preferred path if strategy_features ran)
# -------------------------------------------------------------------
ENTRY_STD_KEYS = [
    "signal_mean_reversion",
    "signal_trend_follow",
    "signal_breakout_buy",
    "signal_breakout_sell",
]

def entry_from_features(df) -> Dict[str, int]:
    """Build standardized entry flags from feature columns if present.
    This reads "signal_*" columns produced by strategy_features or the env.
    Falls back to zeros if columns are missing.
    """
    out = {k: 0 for k in ENTRY_STD_KEYS}
    if df is None or len(df) == 0:
        return out
    try:
        last = df.iloc[-1]
        for k in ENTRY_STD_KEYS:
            if k in df.columns:
                out[k] = int(1 if last[k] else 0)
    except Exception as e:
        logging.debug("[entry_from_features] failed: %s", e)
    return out


# -------------------------------------------------------------------
# Entry signals via original module (kept intact)
# -------------------------------------------------------------------

def entry_from_module(human_names: List[str]) -> Dict[str, int]:
    return map_human_to_standard(human_names)


# -------------------------------------------------------------------
# Danger/Freeze/Recovery adapters
# Expect ctx as a read-only dict with recent indicators or snapshot values.
# The concrete ctx shape is determined by the original modules.
# -------------------------------------------------------------------
try:
    from .signals.danger_signals import compute_danger_signals as _danger
except Exception:
    _danger = None
try:
    from .signals.freeze_signals import compute_freeze_signals as _freeze
except Exception:
    _freeze = None
try:
    from .signals.recovery_signals import compute_recovery_signals as _recovery
except Exception:
    _recovery = None


def _to_flags(d: Any, key_prefix: str) -> Dict[str, int]:
    """Normalize module outputs (list/dict/int) into flat flags with prefix.
    - list -> {f"{key_prefix}_{name}":1}
    - dict(counts) -> {f"{key_prefix}_{k}": int(v>0)}
    - int/bool -> {f"{key_prefix}_any": int(v>0)}
    """
    out: Dict[str, int] = {}
    if d is None:
        return out
    if isinstance(d, dict):
        for k, v in d.items():
            out[f"{key_prefix}_{str(k)}"] = 1 if (int(v) if isinstance(v, (int, float)) else bool(v)) else 0
    elif isinstance(d, (list, tuple)):
        for k in d:
            out[f"{key_prefix}_{str(k)}"] = 1
    elif isinstance(d, (int, float, bool)):
        out[f"{key_prefix}_any"] = 1 if int(d) > 0 else 0
    else:
        logging.debug("[_to_flags] unsupported type: %s", type(d))
    return out


def danger_from_module(ctx: Dict[str, Any]) -> Dict[str, int]:
    if _danger is None:
        return {}
    try:
        d = _danger(ctx)
    except Exception as e:
        logging.debug("[danger_from_module] failed: %s", e)
        return {}
    return _to_flags(d, "danger")


def freeze_from_module(ctx: Dict[str, Any]) -> Dict[str, int]:
    if _freeze is None:
        return {}
    try:
        d = _freeze(ctx)
    except Exception as e:
        logging.debug("[freeze_from_module] failed: %s", e)
        return {}
    return _to_flags(d, "freeze")


def recovery_from_module(ctx: Dict[str, Any]) -> Dict[str, int]:
    if _recovery is None:
        return {}
    try:
        d = _recovery(ctx)
    except Exception as e:
        logging.debug("[recovery_from_module] failed: %s", e)
        return {}
    return _to_flags(d, "recovery")


# -------------------------------------------------------------------
# Reward utilities
# -------------------------------------------------------------------

def reward_blend(base_reward: float, signals: Dict[str, float], weights: Dict[str, float]) -> float:
    """Blend base reward with (signals * weights) safely.
    Any missing keys are treated as zero. The function is pure.
    """
    try:
        extra = 0.0
        for k, w in (weights or {}).items():
            v = float(signals.get(k, 0.0))
            extra += float(w) * v
        return float(base_reward) + extra
    except Exception:
        return float(base_reward)


__all__ = [
    "map_human_to_standard",
    "entry_from_features",
    "entry_from_module",
    "danger_from_module",
    "freeze_from_module",
    "recovery_from_module",
    "reward_blend",
    "HUMAN2STD",
]
