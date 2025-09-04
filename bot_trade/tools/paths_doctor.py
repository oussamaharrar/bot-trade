"""Check run directory structure and symlinks."""
from __future__ import annotations

import argparse
from pathlib import Path

from bot_trade.tools.force_utf8 import force_utf8

REQUIRED_FILES = {"metrics.csv", "summary.json", "risk_flags.jsonl"}
REQUIRED_DIRS = {"charts", "artifacts"}


def _hint(name: str, run: Path) -> str:
    cmd = f"python -m bot_trade.eval.eval_run --run-dir {run}" if name != "best_model.zip" else "python bot_trade/train_rl.py <args>"
    return f"Run Train/Eval first: {cmd}"


def _check_run(run: Path) -> bool:
    ok = True
    for name in REQUIRED_FILES:
        if not (run / name).exists():
            print(f"[PATHS] missing {name} in {run} -> {_hint(name, run)}")
            ok = False
    charts = run / "charts"
    if not charts.exists() or not any(charts.glob("*.png")):
        print(f"[PATHS] missing charts/*.png in {run} -> {_hint('charts', run)}")
        ok = False
    artifacts = run / "artifacts" / "best_model.zip"
    if not artifacts.exists():
        print(f"[PATHS] missing artifacts/best_model.zip in {run} -> {_hint('best_model.zip', run)}")
        ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate results layout")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--strict", action="store_true")
    ns = ap.parse_args(argv)

    run = Path("results") / ns.symbol / ns.frame / ns.run_id
    if run.is_symlink() or ns.run_id == "latest":
        if run.exists():
            run = run.resolve()
    if not run.exists():
        print(f"[PATHS] run directory not found: {run}")
        return 2 if ns.strict else 0

    ok = _check_run(run)
    print(f"[PATHS] strict={ns.strict} ok={ok}")
    return 0 if ok or not ns.strict else 2


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
