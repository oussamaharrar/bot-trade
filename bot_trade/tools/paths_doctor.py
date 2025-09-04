"""Check run directory structure and symlinks."""
from __future__ import annotations

import argparse
from pathlib import Path

from bot_trade.tools.force_utf8 import force_utf8

REQUIRED_FILES = {"metrics.csv", "summary.json", "risk_flags.jsonl"}
REQUIRED_DIRS = {"logs", "charts", "artifacts"}


def _check_run(run: Path) -> bool:
    ok = True
    for name in REQUIRED_FILES:
        if not (run / name).exists():
            print(f"[PATHS] missing {name} in {run}")
            ok = False
    for name in REQUIRED_DIRS:
        if not (run / name).exists():
            print(f"[PATHS] missing {name}/ in {run}")
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate results layout")
    ap.add_argument("--strict", action="store_true")
    ns = ap.parse_args(argv)

    root = Path("results")
    if not root.exists():
        print("[PATHS] no results directory")
        return 1

    ok = True
    for sym in root.iterdir():
        for frame in sym.iterdir():
            latest = frame / "latest"
            if latest.exists():
                run_dir = latest.resolve()
                if not _check_run(run_dir):
                    ok = False
            for run in frame.iterdir():
                if run.is_dir() and run.name != "latest":
                    _check_run(run)

    print(f"[PATHS] strict={ns.strict} ok={ok}")
    return 0 if ok or not ns.strict else 1


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
