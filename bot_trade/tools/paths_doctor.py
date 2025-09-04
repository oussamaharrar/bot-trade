"""Check run directory structure and symlinks."""
from __future__ import annotations

import argparse
from pathlib import Path

from bot_trade.tools.force_utf8 import force_utf8

REQUIRED_FILES = {"metrics.csv", "summary.json", "risk_flags.jsonl"}


def _remediation(symbol: str, frame: str) -> None:
    print(
        "Run: python -m bot_trade.tools.gen_synth_data --symbol "
        f"{symbol} --frame {frame} --days 1 --out data_ready"
    )
    print(
        "Then: python bot_trade/train_rl.py --config "
        f"config/train_{symbol.lower()}_{frame}.yaml --max-steps 4000 --seed 7 --data-dir data_ready"
    )
    print(
        "Then: python -m bot_trade.eval.eval_run --run-dir "
        f"results/{symbol}/{frame}/latest --tearsheet-html"
    )


def _check_run(run: Path, symbol: str, frame: str) -> bool:
    ok = True
    for name in REQUIRED_FILES:
        if not (run / name).exists():
            print(f"[PATHS] missing {name} in {run}")
            ok = False
    charts = run / "charts"
    if not charts.exists() or not any(charts.glob("*.png")):
        print(f"[PATHS] missing charts/*.png in {run}")
        ok = False
    artifacts = run / "artifacts" / "best_model.zip"
    if not artifacts.exists():
        print(f"[PATHS] missing artifacts/best_model.zip in {run}")
        ok = False
    if not ok:
        _remediation(symbol, frame)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate results layout",
        epilog="Example: python -m bot_trade.tools.paths_doctor --symbol BTCUSDT --frame 1m --strict",
    )
    ap.add_argument("--symbol", required=True, help="Trading symbol")
    ap.add_argument("--frame", required=True, help="Time frame, e.g. 1m")
    ap.add_argument("--run-id", default="latest", help="Run id or 'latest'")
    ap.add_argument("--strict", action="store_true", help="Exit 2 if missing")
    ns, extras = ap.parse_known_args(argv)
    if extras:
        print(f"[WARN] ignoring extras: {' '.join(extras)}")

    run = Path("results") / ns.symbol / ns.frame / ns.run_id
    if run.is_symlink() or ns.run_id == "latest":
        if run.exists():
            run = run.resolve()
    if not run.exists():
        print(f"[PATHS] run directory not found: {run}")
        _remediation(ns.symbol, ns.frame)
        return 2 if ns.strict else 0

    ok = _check_run(run, ns.symbol, ns.frame)
    print(f"[PATHS] strict={ns.strict} ok={ok}")
    return 0 if ok or not ns.strict else 2


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
