from __future__ import annotations

"""Utility for appending run metadata to the knowledge base.

This module centralizes writes to the Knowlogy knowledge base. Entries are
stored in JSON Lines format with one JSON object per line.  Writes are
performed atomically to avoid corrupting the file.
"""

from pathlib import Path
import json
import os
from typing import Any, Mapping

from bot_trade.config.rl_paths import DEFAULT_KB_FILE


def _resolve_kb_path(run_paths: Any) -> Path:
    """Return KB path from a RunPaths instance or mapping.

    Falls back to the canonical location when ``kb_file`` is missing.
    """

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


def kb_append(run_paths: Any, payload: dict) -> None:
    """Atomically append ``payload`` to the knowledge base.

    Parameters
    ----------
    run_paths:
        RunPaths instance, mapping, or path-like pointing to the KB file.
    payload:
        Dictionary payload to serialize as JSON.
    """

    path = _resolve_kb_path(run_paths)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except Exception:
            existing = ""

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        if existing:
            fh.write(existing.rstrip("\n") + "\n")
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
