"""Run context helpers: run id generation and filesystem utilities."""
from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from bot_trade.tools.paths import ROOT
from bot_trade.config.rl_paths import build_paths, new_run_id as _new_run_id
from bot_trade.tools.atomic_io import write_json


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
        return out
    except Exception:
        return "nogit"


def new_run_id(symbol: str, frame: str) -> str:  # noqa: ARG001
    """Return a short unique run identifier."""

    # Historical interface kept for compatibility; symbol/frame are ignored.
    return _new_run_id()


def run_paths(symbol: str, frame: str, run_id: str) -> Dict[str, Path]:
    p = build_paths(symbol, frame, run_id)
    return {
        "results": Path(p["results"]),
        "report": Path(p["reports"]),
        "logs": Path(p["logs"]),
        "agents": Path(p["agents"]),
    }


def atomic_write_json(path: Path, data: Dict) -> None:
    """[DEPRECATED] use :func:`bot_trade.tools.atomic_io.write_json`."""
    write_json(path, data)


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
