"""Run context helpers: run id generation and atomic filesystem utils."""
from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from bot_trade.tools.paths import ROOT
from bot_trade.config.rl_paths import build_paths


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
        return out
    except Exception:
        return "nogit"


def new_run_id(symbol: str, frame: str) -> str:
    """Return a session-stable run identifier.

    Format: ``run-<SYMBOL>-<FRAME>-<YYYYMMDD_HHMMSS>-<shortid>``.
    ``shortid`` is the first 4 hex chars of the current git hash or ``nogit``.

    # TODO: consider allowing custom run id prefix/suffix via CLI.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gh = _git_hash()[:4]
    return f"run-{symbol}-{frame}-{ts}-{gh}"


def run_paths(symbol: str, frame: str, run_id: str) -> Dict[str, Path]:
    p = build_paths(symbol, frame, run_id)
    return {
        "results": Path(p["results"]),
        "report": Path(p["reports"]),
        "logs": Path(p["logs"]),
        "agents": Path(p["agents"]),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: Dict) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


@contextmanager
def lockfile(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield fh
        finally:
            if os.name == "posix":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
