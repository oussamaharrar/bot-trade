"""Run monitor/exporter for training charts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bot_trade.config.rl_paths import RunPaths, get_root, DEFAULT_REPORTS_DIR
from bot_trade.tools.latest import latest_run
from bot_trade.tools import export_charts
from bot_trade.tools._headless import ensure_headless_once


def main(argv: list[str] | None = None) -> int:
    ensure_headless_once("tools.monitor_manager")
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
    ap.add_argument("--tearsheet", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--pdf", action="store_true", help=argparse.SUPPRESS)
    ns = ap.parse_args(argv)

    try:
        root = Path(ns.base).resolve() if ns.base else get_root()
        reports_root = (
            Path(ns.base).resolve() / "reports" if ns.base else Path(DEFAULT_REPORTS_DIR)
        )
        reports_root = reports_root / "PPO"
        run_id = ns.run_id
        if run_id in {None, "latest", "", "last"}:
            run_id = latest_run(ns.symbol, ns.frame, reports_root)
            if not run_id:
                print("[LATEST] none")
                return 2
        rp = RunPaths(ns.symbol, ns.frame, run_id, root=root)
        charts_dir, images, rows = export_charts.export_run_charts(
            rp, run_id, debug=ns.debug_export
        )
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
        if ns.tearsheet:
            from bot_trade.eval.tearsheet import generate_tearsheet

            generate_tearsheet(rp, pdf=ns.pdf)
        return 0 if images > 0 else 2
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

