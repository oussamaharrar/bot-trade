"""Unified memory manager for resumable training and analysis."""
from __future__ import annotations

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from bot_trade.config.rl_paths import get_root, memory_dir
from bot_trade.tools.paths import ensure_dirs, DIR_MEMORY
from bot_trade.tools.runctx import atomic_write_json, lockfile


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryManager:
    """Append-only event log with periodic state snapshots."""

    def __init__(self, base_dir: Path = DIR_MEMORY) -> None:
        self.base_dir = base_dir
        ensure_dirs(self.base_dir)
        self.events_file = self.base_dir / "events.jsonl"
        self.state_file = self.base_dir / "state_latest.json"
        self.index_file = self.base_dir / "state_index.jsonl"

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        entry = {"type": event_type, "ts": _ts(), **payload}
        lock = self.events_file.with_suffix(".lock")
        ensure_dirs(self.base_dir)
        with lockfile(lock):
            with self.events_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")

    def snapshot(self, state: Dict[str, Any]) -> None:
        data = {"updated_at": _ts(), **state}
        atomic_write_json(self.state_file, data)

    def resume(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with self.state_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def compact(self) -> None:
        size = self.events_file.stat().st_size if self.events_file.exists() else 0
        entry = {"ts": _ts(), "size": size}
        lock = self.index_file.with_suffix(".lock")
        with lockfile(lock):
            with self.index_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")


# =============================
# Memory v2 helpers
# =============================

PROJECT_ROOT = get_root()
MEM_DIR = memory_dir()
MEMORY_FILE = MEM_DIR / "memory.json"


def _default_memory(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "root": str(root),
        "last_run_id": None,
        "runs": {},
        "frames": {},
        "sessions": {},
    }


def load_memory(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Load memory.json upgrading to schema v2 if needed."""
    path = (memory_dir() if root == PROJECT_ROOT else Path(root) / "memory") / "memory.json"
    if not path.exists():
        return _default_memory(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_memory(root)

    # v1 -> v2 migration: v1 did not have schema_version and stored
    # simple keys like ``run_id``/``step``.  Merge anything we can find
    # into the new structure while keeping defaults for missing fields.
    if data.get("schema_version") != 2:
        upgraded = _default_memory(root)
        # try legacy root keys
        if isinstance(data, dict):
            if "last_run_id" in data:
                upgraded["last_run_id"] = data.get("last_run_id")
            if "runs" in data:
                upgraded["runs"] = data.get("runs", {})
            if "frames" in data:
                upgraded["frames"] = data.get("frames", {})
            if "sessions" in data:
                upgraded["sessions"] = data.get("sessions", {})
            # very old memory file: {run_id:..., step:...}
            if "run_id" in data and "step" in data:
                rid = str(data.get("run_id"))
                upgraded["last_run_id"] = rid
                upgraded.setdefault("runs", {})[rid] = {
                    "global": {"global_timesteps": int(data.get("step", 0))},
                    "updated_at": data.get("updated_at"),
                }
        return upgraded

    return data


def validate_dataset(info: Dict[str, Any]) -> bool:
    """Validate dataset info against current file stats."""
    path = info.get("path")
    if not path:
        return True
    p = Path(path)
    if not p.exists():
        return False
    try:
        st = p.stat()
        if int(st.st_mtime) != int(info.get("mtime", st.st_mtime)):
            return False
        if int(st.st_size) != int(info.get("size_bytes", st.st_size)):
            return False
    except Exception:
        return False
    return True


def make_snapshot(
    args: Any,
    env: Any,
    model: Any,
    vecnorm: Any,
    writers: Any,
    risk_manager: Any,
    dataset_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a tolerant snapshot dictionary."""

    def _try(obj, attr, default=None):
        try:
            return getattr(obj, attr)
        except Exception:
            return default

    snapshot: Dict[str, Any] = {
        "meta": {
            "seed": getattr(args, "seed", None),
            "device": getattr(args, "device_str", getattr(args, "device", None)),
            "n_envs": getattr(args, "n_envs", None),
            "algo": getattr(args, "algo", "PPO"),
            "policy": getattr(args, "policy", getattr(args, "policy_name", None)),
        },
        "dataset": dataset_info or {},
        "global": {
            "global_timesteps": _try(model, "num_timesteps", 0),
            "vecnorm_path": _try(vecnorm, "save_path", None),
            "checkpoint_path": _try(model, "_save_path", None),
            "best_path": _try(getattr(writers, "train", None), "best_path", None),
        },
        "env_state": {},
        "risk_state": {},
        "writers": {
            "last_artifact_step": _try(getattr(writers, "train", None), "last_step", 0),
            "last_flush_at": datetime.now(timezone.utc).isoformat(),
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # meta extras
    snapshot["meta"].update({
        "symbol": getattr(args, "symbol", None),
        "frame": getattr(args, "frame", None),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    # attempt to pull common env fields
    env_state = snapshot["env_state"]
    try:
        env_state["ptr_index"] = int(_try(env, "ptr", 0))
    except Exception:
        pass
    try:
        env_state["ptr_ts"] = _try(env, "current_ts", None)
    except Exception:
        try:
            env_state["ptr_ts"] = _try(_try(env, "_row", lambda: None)(), "name", None)
        except Exception:
            pass
    try:
        import numpy as np, random
        env_state["rng_numpy"] = list(np.random.get_state())  # type: ignore[arg-type]
        env_state["rng_python"] = list(random.getstate())  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        pos = {
            "side": _try(env, "position_side", None),
            "size": _try(env, "coin", None),
            "entry": _try(env, "entry_price", None),
            "sl": _try(env, "stop_loss_price", None),
            "tp": _try(env, "take_profit_price", None),
        }
        if any(v is not None for v in pos.values()):
            env_state["open_position"] = pos
    except Exception:
        pass
    for k in ["equity", "pnl_real", "pnl_unreal", "last_action", "last_reward", "last_reward_components"]:
        try:
            env_state[k] = _try(env, k, None)
        except Exception:
            pass

    # risk manager state
    rstate = snapshot["risk_state"]
    for attr, key in [
        ("current_risk", "risk_pct"),
        ("danger_mode", "danger_mode"),
        ("freeze_mode", "freeze_mode"),
        ("ema_reward", "ema_reward"),
        ("max_drawdown", "max_drawdown"),
    ]:
        try:
            rstate[key] = _try(risk_manager, attr, None)
        except Exception:
            pass

    return snapshot


def commit_snapshot(run_id: str, snapshot: Dict[str, Any], root: Path = PROJECT_ROOT) -> None:
    """Atomically commit ``snapshot`` for ``run_id`` to memory.json."""
    mem = load_memory(root)
    mem.setdefault("runs", {})[run_id] = snapshot
    mem["last_run_id"] = run_id
    path = (memory_dir() if root == PROJECT_ROOT else Path(root) / "memory") / "memory.json"
    ensure_dirs(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def resume_from_snapshot(run_id: str, root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Return stored snapshot for ``run_id`` ensuring dataset still matches."""
    mem = load_memory(root)
    snap = mem.get("runs", {}).get(run_id)
    if not snap:
        return {}
    if not validate_dataset(snap.get("dataset", {})):
        raise SystemExit(f"[MEMORY] Dataset mismatch for run {run_id}; aborting resume.")
    return snap


__all__ = [
    "MemoryManager",
    "load_memory",
    "validate_dataset",
    "make_snapshot",
    "commit_snapshot",
    "resume_from_snapshot",
]

