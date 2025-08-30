# ai_core/self_improver.py
# Adapt config using both KB (long-term) and latest session analytics
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import json
import os
import yaml

from bot_trade.config.rl_paths import ensure_utf8, memory_dir

KB_PATH = memory_dir() / "knowledge_base_full.json"
CONFIG_PATH = os.getenv("BOT_CONFIG", os.path.join("config", "config.yaml"))
SUCCESS_PATH = memory_dir() / "success_patterns.json"
FAILURE_PATH = memory_dir() / "failure_insights.json"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        with ensure_utf8(path, csv_newline=False) as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    try:
        with ensure_utf8(path, csv_newline=False) as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    except Exception:
        pass


def propose_config_updates() -> Dict[str, Any]:
    """Compute safe config updates based on KB+recent analytics."""
    kb = _load_json(KB_PATH)
    succ = _load_json(SUCCESS_PATH)
    fail = _load_json(FAILURE_PATH)

    updates: Dict[str, Any] = {"_meta": {"ts": datetime.utcnow().isoformat()}}

    # Example 1: if a frame has poor mean reward → lower risk multiplier for that frame
    strat = kb.get("strategy_memory", {})
    bad_frames = [k for k, v in strat.items() if float(v.get("reward_mean", 0.0)) < 0.0]
    if bad_frames:
        updates.setdefault("risk", {})
        updates["risk"].setdefault("frame_overrides", {})
        for fr in bad_frames:
            updates["risk"]["frame_overrides"][str(fr)] = {"risk_multiplier": 0.5}

    # Example 2: if a signal has low win-rate → mark it as risky (for entry_verifier to filter)
    sigs = kb.get("signals_memory", {})
    risky_sigs = [k for k, v in sigs.items() if float(v.get("win_rate", 0.5)) < 0.35 and float(v.get("count", 0)) >= 50]
    if risky_sigs:
        updates.setdefault("policy", {})
        existing = set(kb.get("policy", {}).get("ban_signals", []) or [])
        updates["policy"]["ban_signals"] = sorted(list(existing.union(risky_sigs)))

    # Example 3: integrate quick success/failure hints (if present)
    if succ.get("prefer_spread", None) is True:
        updates.setdefault("trade", {})
        updates["trade"]["spread"] = max(0.0, float(succ.get("spread", 0.0)))
    if fail.get("slippage_too_high", None) is True:
        updates.setdefault("trade", {})
        updates["trade"]["slippage"] = max(0.0, float(fail.get("slippage", 0.0)))

    return updates


def apply_updates_to_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    if not updates:
        return {}
    cfg = _load_yaml(CONFIG_PATH)
    # shallow merge (safe)
    for k, v in updates.items():
        if k == "_meta":
            continue
        if isinstance(v, dict):
            node = cfg.get(k, {}) if isinstance(cfg.get(k), dict) else {}
            node.update(v)
            cfg[k] = node
        else:
            cfg[k] = v
    _save_yaml(CONFIG_PATH, cfg)
    return cfg


if __name__ == "__main__":
    ups = propose_config_updates()
    new_cfg = apply_updates_to_config(ups)
    print("Applied updates to config:", list(ups.keys()))
