from __future__ import annotations

"""Simple reconciliation helper."""

from pathlib import Path
import json
import time


def reconcile_positions(perf_dir: Path, ledger: dict, gateway) -> None:  # pragma: no cover - simple stub
    line = json.dumps({"ts": time.time(), "ledger": ledger})
    path = perf_dir / "reconcile.jsonl"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(line + "\n", encoding="utf-8")
    tmp.replace(path)
