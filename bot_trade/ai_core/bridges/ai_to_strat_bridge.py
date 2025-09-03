from __future__ import annotations
from pathlib import Path
from typing import Iterable

from bot_trade.tools.atomic_io import append_jsonl
from bot_trade.config.rl_paths import memory_dir


def write_signals(records: Iterable[dict], dry_run: bool = False) -> Path:
    """Atomically append signal records to memory/Knowlogy/signals.jsonl."""
    path = memory_dir() / "Knowlogy" / "signals.jsonl"
    if dry_run:
        return path
    for rec in records:
        append_jsonl(path, rec)
    return path
