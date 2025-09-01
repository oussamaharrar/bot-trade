from __future__ import annotations

"""Utility for appending run metadata to the knowledge base.

This module centralizes writes to the Knowlogy knowledge base. Entries are
stored in JSON Lines format with one JSON object per line.  Writes are
performed atomically to avoid corrupting the file.
"""

from pathlib import Path
import json
import os
from typing import Any, Mapping, Optional

from bot_trade.config.rl_paths import DEFAULT_KB_FILE


def _resolve_kb_path(run_paths: Any, kb_file: Optional[str] = None) -> Path:
    """Return KB path derived from ``run_paths`` or explicit override."""

    if kb_file:
        return Path(kb_file)
    if isinstance(run_paths, (str, Path)):
        return Path(run_paths)
    if isinstance(run_paths, Mapping):
        kb = run_paths.get("kb_file")
        if kb:
            return Path(kb)
    kb = getattr(run_paths, "kb_file", None)
    if kb:
        return Path(kb)
    return Path(DEFAULT_KB_FILE)


def kb_append(run_paths: Any, payload: dict, kb_file: Optional[str] = None) -> None:
    """Append ``payload`` as JSON to the knowledge base.

    Writes are performed using ``os.O_APPEND`` to avoid clobbering existing
    data. The file is created if missing and always written with a trailing
    newline to maintain JSON Lines format.
    """

    path = _resolve_kb_path(run_paths, kb_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(payload, ensure_ascii=False) + "\n"
    data = line.encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "ab") as fh:
        fh.write(data)
