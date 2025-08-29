"""Simple persistent memory store with atomic updates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from tools.paths import DIR_MEMORY, ensure_dirs
from tools.runctx import atomic_write_json

MEMORY_FILE = DIR_MEMORY / "memory.json"


def load() -> Dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save(data: Dict) -> None:
    ensure_dirs(DIR_MEMORY)
    atomic_write_json(MEMORY_FILE, data)


def update(partial: Dict) -> Dict:
    data = load()
    data.update(partial)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(data)
    return data


def checkpoint(run_id: str, step: int, extra: Dict | None = None) -> Dict:
    payload = {"run_id": run_id, "step": step}
    if extra:
        payload.update(extra)
    return update(payload)
