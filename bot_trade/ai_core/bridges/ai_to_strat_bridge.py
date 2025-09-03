from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Set, Tuple

from bot_trade.tools.atomic_io import append_jsonl
from bot_trade.config.rl_paths import memory_dir

_WRITE_WARNED = False


def write_signals(records: Iterable[dict], dry_run: bool = False) -> Path:
    """Atomically append signal records to memory/Knowlogy/signals.jsonl."""
    path = memory_dir() / "Knowlogy" / "signals.jsonl"
    if dry_run:
        return path
    seen: Set[Tuple[str, str, str]] = set()
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        key = (obj.get("ts"), obj.get("symbol"), obj.get("signal"))
                        seen.add(key)
                    except Exception:
                        continue
    except Exception:
        pass
    try:
        for rec in records:
            key = (rec.get("ts"), rec.get("symbol"), rec.get("signal"))
            if key in seen:
                continue
            seen.add(key)
            append_jsonl(path, rec)
    except Exception as e:
        global _WRITE_WARNED
        if not _WRITE_WARNED:
            print(f"[AI_CORE] write_failed={e}")
            _WRITE_WARNED = True
    return path
