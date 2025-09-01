"""Run monitor/exporter for training charts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from bot_trade.config.rl_paths import RunPaths, get_root


def _latest_run(symbol: str, frame: str, reports_root: Path) -> str | None:
    base = reports_root / symbol / frame
    if not base.exists():
        return None
    run_dirs = [d for d in base.iterdir() if d.is_dir() and not d.is_symlink()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate charts for a finished training run",
        epilog=(
            "Example: python -m bot_trade.tools.monitor_manager "
            "--symbol BTCUSDT --frame 1m --run-id latest"
        ),
    )
    ap.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    ap.add_argument("--frame", default="1m", help="Time frame")
    ap.add_argument("--run-id", default="latest", help="Run identifier or 'latest'")
    ap.add_argument("--base", default=None, help="Project root override")
    ap.add_argument("--debug-export", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-wait", action="store_true", help=argparse.SUPPRESS)
    ns = ap.parse_args(argv)

    try:
        root = Path(ns.base) if ns.base else get_root()
        reports_root = root / "reports"
        run_id = ns.run_id
        if run_id in {None, "latest", ""}:
            run_id = _latest_run(ns.symbol, ns.frame, reports_root)
            if not run_id:
                print("[LATEST] none")
                return 2
        rp = RunPaths(ns.symbol, ns.frame, run_id, root=root)
        from bot_trade.tools import export_run_charts  # lazy heavy import

        charts_dir, images, rows = export_run_charts.export_for_run(rp)
        print(
            "[DEBUG_EXPORT] reward_rows=%d step_rows=%d train_rows=%d risk_rows=%d callbacks_rows=%d signals_rows=%d"
            % (
                rows.get("reward", 0),
                rows.get("step", 0),
                rows.get("train", 0),
                rows.get("risk", 0),
                rows.get("callbacks", 0),
                rows.get("signals", 0),
            )
        )
        print(f"[CHARTS] dir={charts_dir.resolve()} images={images}")
        return 0 if images > 0 else 2
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

