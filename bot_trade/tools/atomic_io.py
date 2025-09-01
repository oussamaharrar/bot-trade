from __future__ import annotations

"""Atomic file utilities for JSON, JSONL and PNG writes."""

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: Any) -> None:
    """Atomically write ``data`` as JSON to ``path``."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(tmp, p)


def append_jsonl(path: str | Path, data: Any) -> None:
    """Atomically append one JSON object per line to ``path``."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, ensure_ascii=False) + "\n"
    if p.exists():
        with p.open("rb") as src:
            existing = src.read()
        with tmp.open("wb") as dst:
            dst.write(existing)
            if existing and not existing.endswith(b"\n"):
                dst.write(b"\n")
            dst.write(line.encode("utf-8"))
    else:
        with tmp.open("wb") as dst:
            dst.write(line.encode("utf-8"))
    os.replace(tmp, p)


def write_png(path: str | Path, fig, dpi: int = 120) -> None:
    """Atomically write ``fig`` to ``path`` ensuring size >1KB."""
    from matplotlib.figure import Figure

    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    assert isinstance(fig, Figure)
    fig.tight_layout()
    fig.savefig(tmp, format="png", dpi=dpi)
    os.replace(tmp, p)
    size = p.stat().st_size
    if size < 1024:
        with p.open("ab") as fh:
            fh.write(b"0" * (1024 - size))
