from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict

from bot_trade.config.rl_paths import memory_dir
from bot_trade.tools.atomic_io import write_json

RUN_STATE_PATH = memory_dir() / "run_state.json"


def load_state(path: Path = RUN_STATE_PATH) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict, path: Path = RUN_STATE_PATH) -> None:
    write_json(path, state)


def update_portfolio_state(path: Path, steps: int) -> None:
    state = {"equity": 0.0, "steps": 0, "last_ts": "", "version": 1}
    if path.exists():
        try:
            state.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    state["steps"] = int(state.get("steps", 0)) + int(steps)
    state["last_ts"] = dt.datetime.utcnow().isoformat()
    write_json(path, state)


def write_run_state_files(perf_dir: Path, run_id: str, create_lock: bool = False) -> None:
    perf_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.utcnow().isoformat()
    write_json(perf_dir / "run_state.json", {"status": "idle", "updated_at": now})
    write_json(perf_dir / "state_latest.json", {"last_run_id": run_id, "updated_at": now})
    if create_lock:
        (perf_dir / "events.lock").touch()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ns = ap.parse_args()
    if ns.resume:
        st = load_state()
        print(json.dumps(st, indent=2) if st else "{}")
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
