from __future__ import annotations

"""Helpers for resolving latest run directories."""

from pathlib import Path
from typing import Optional

from bot_trade.config.rl_paths import DEFAULT_REPORTS_DIR


def latest_run(symbol: str, frame: str, reports_root: str | Path = DEFAULT_REPORTS_DIR) -> Optional[str]:
    base = Path(reports_root) / symbol / frame
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir() and not d.is_symlink()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0].name
