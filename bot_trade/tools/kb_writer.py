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


KB_DEFAULTS = {
    "run_id": "",
    "symbol": "",
    "frame": "",
    "ts": None,
    "images": 0,
    "rows_reward": 0,
    "rows_step": 0,
    "rows_train": 0,
    "rows_risk": 0,
    "rows_callbacks": 0,
    "rows_signals": 0,
    "vecnorm_applied": False,
    "vecnorm_snapshot_saved": False,
    "best": False,
    "last": False,
    "best_model_path": "",
    "eval": {
        "win_rate": None,
        "sharpe": None,
        "max_drawdown": None,
        "avg_trade_pnl": None,
    },
    "portfolio": {
        "equity": None,
        "cash": None,
        "positions": None,
        "step": None,
    },
    "notes": "",
}


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

    entry = {
        **KB_DEFAULTS,
        **{k: v for k, v in payload.items() if k not in {"eval", "portfolio"}},
    }
    entry["eval"] = {**KB_DEFAULTS["eval"], **payload.get("eval", {})}
    entry["portfolio"] = {**KB_DEFAULTS["portfolio"], **payload.get("portfolio", {})}

    # prevent duplicate run_id appends
    if path.exists():
        try:
            with path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(size - 4096, 0))
                data = fh.read()
            last_line = data.splitlines()[-1] if data else b""
            last = json.loads(last_line.decode("utf-8")) if last_line else {}
            if last.get("run_id") == entry.get("run_id"):
                return
        except Exception:
            pass

    line = json.dumps(entry, ensure_ascii=False) + "\n"
    data = line.encode("utf-8")
    prefix = b""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                prefix = b"\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "ab") as fh:
        if prefix:
            fh.write(prefix)
        fh.write(data)
