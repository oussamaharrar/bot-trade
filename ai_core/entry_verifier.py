# ai_core/entry_verifier.py
# 🧠 Purpose: Prevent risky entries using knowledge stats + pre-trade simulation
# - Unified KB path (knowledge_base_full.json)
# - Robust logging & error handling
# - Works with config/signals_bridge for mapping signals
# - Safe defaults when config/KB is missing

from __future__ import annotations
import json, os, traceback
from typing import Any, Dict, Iterable, List, Optional

try:
    from config.signals_bridge import map_human_to_standard
except Exception:
    # Fallback no-op mapping
    def map_human_to_standard(names: List[str]) -> Dict[str, int]:
        return {str(n): 1 for n in (names or [])}

try:
    from ai_core.simulation_engine import simulate_entry
except Exception:
    # Minimal fallback: always neutral
    def simulate_entry(close_series, signal_time_idx: int, **kwargs) -> int:
        return 0

KB_FILE = "memory/knowledge_base_full.json"

# ---------------------------
# KB helpers
# ---------------------------

def _load_json(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------
# Policy derived from KB
# ---------------------------

def _disallow_map(kb: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict of disallowed frames/symbols/signals from KB (safe defaults)."""
    pol = kb.get("policy", {}) if isinstance(kb, dict) else {}
    return {
        "frames": set(pol.get("ban_frames", []) or []),
        "symbols": set(pol.get("ban_symbols", []) or []),
        "signals": set(pol.get("ban_signals", []) or []),
    }


# ---------------------------
# Public API
# ---------------------------

def filter_signals(human_names: Iterable[str], *, frame: str, symbol: str) -> List[str]:
    """Filter human signal names against KB bans and return ALLOWED human names.
    The env will map them to standard keys via signals_bridge.
    """
    names = [str(x).strip() for x in (human_names or []) if str(x).strip()]
    if not names:
        return []
    kb = _load_json(KB_FILE)
    bans = _disallow_map(kb)
    std = map_human_to_standard(names)
    # If any standard key is banned → drop its human name
    allowed: List[str] = []
    for n in names:
        k = map_human_to_standard([n]).keys()
        if not k:
            allowed.append(n)  # unknown name -> allow (no hard bans)
            continue
        key = next(iter(k))
        if key in bans["signals"]:
            continue
        allowed.append(n)
    # Frame/Symbol wide ban
    if frame in bans["frames"] or symbol in bans["symbols"]:
        return []
    return allowed


def smart_entry_guard(
    frame: str,
    symbol: str,
    *,
    df_slice,
    signal_idx: int,
    active_signals: Iterable[str],
    min_rr: float = 1.2,
    spread: float = 0.0,
    slippage: float = 0.0,
    lookahead: int = 100,
) -> bool:
    """Return True if entry is allowed under KB policy + quick simulation.

    Heuristics:
    1) If frame/symbol globally banned in KB → deny.
    2) Drop signals banned in KB → if nothing remains → deny.
    3) Run a quick rule-based simulation around `signal_idx`:
       - If expected RR < min_rr → deny; if TP hit before SL → allow; else neutral → allow conservatively.
    """
    kb = _load_json(KB_FILE)
    bans = _disallow_map(kb)
    if frame in bans["frames"] or symbol in bans["symbols"]:
        return False

    allowed_human = filter_signals(active_signals, frame=frame, symbol=symbol)
    if not allowed_human:
        return False

    # Quick simulation (close-only). The engine returns {-1,0,1}
    try:
        close = df_slice["close"].astype("float32").values
    except Exception:
        # Try generic sequence-like
        close = getattr(df_slice, "values", df_slice)
    try:
        outcome = simulate_entry(
            close,
            signal_time_idx=signal_idx,
            lookahead=lookahead,
            spread=spread,
            slippage=slippage,
            min_rr=min_rr,
        )
    except Exception:
        outcome = 0

    if outcome < 0:
        return False
    return True


# ---------------------------
# CLI quick test
# ---------------------------
if __name__ == "__main__":
    import pandas as pd, numpy as np
    # Fake data
    prices = 100 + np.cumsum(np.random.randn(500).astype("float32"))
    df = pd.DataFrame({"close": prices})
    ok = smart_entry_guard("1m", "BTCUSDT", df_slice=df, signal_idx=250, active_signals=["RSI recovery", "MACD bullish"])
    print("ALLOWED?", ok)
