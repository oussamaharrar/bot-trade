"""Unified memory manager for resumable training and analysis."""
from __future__ import annotations

from tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from tools.paths import DIR_MEMORY, ensure_dirs
from tools.runctx import atomic_write_json, lockfile


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
                fh.write(json.dumps(entry) + "\n")

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
