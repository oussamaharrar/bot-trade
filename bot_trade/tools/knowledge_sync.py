"""Persist run knowledge summaries.

This module provides a small helper to record knowledge artifacts for a run
so future sessions can learn from previous executions. It writes a detailed
JSON file for each run and appends a line to ``memory/knowledge/log.jsonl``.
"""
from __future__ import annotations

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from bot_trade.config.rl_paths import ensure_utf8, memory_dir
from bot_trade.tools.paths import ensure_dirs
from bot_trade.tools.runctx import atomic_write_json


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_knowledge(run_id: str, summary: str, *, references: Iterable[str] | None = None,
                    strategies: Iterable[str] | None = None,
                    hints: Iterable[str] | None = None,
                    failures: Iterable[str] | None = None,
                    next_steps: Iterable[str] | None = None) -> Path:
    """Write structured knowledge for ``run_id``.

    A compact dictionary is stored to ``memory/knowledge/knowledge-{run_id}.json``
    and an entry is appended to ``memory/knowledge/log.jsonl``.
    """
    dir_knowledge = memory_dir() / "knowledge"
    ensure_dirs(dir_knowledge)
    data: Dict[str, object] = {
        "run_id": run_id,
        "summary": summary,
        "strategies": list(strategies or []),
        "hints": list(hints or []),
        "failures": list(failures or []),
        "next_steps": list(next_steps or []),
        "references": list(references or []),
        "created_at": _ts(),
    }
    path = dir_knowledge / f"knowledge-{run_id}.json"
    atomic_write_json(path, data)
    log_path = dir_knowledge / "log.jsonl"
    with ensure_utf8(log_path, csv_newline=False) as fh:
        fh.write(json.dumps(data) + "\n")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Record a knowledge summary")
    ap.add_argument("run_id")
    ap.add_argument("summary")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--strategy", action="append", default=[])
    ap.add_argument("--hint", action="append", default=[])
    ap.add_argument("--failure", action="append", default=[])
    ap.add_argument("--next", action="append", dest="next_step", default=[])
    args = ap.parse_args()

    write_knowledge(
        args.run_id,
        args.summary,
        references=args.ref,
        strategies=args.strategy,
        hints=args.hint,
        failures=args.failure,
        next_steps=args.next_step,
    )
