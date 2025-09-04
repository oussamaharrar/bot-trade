"""Thin wrapper around :mod:`bot_trade.tools.eval_run` for unified CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from bot_trade.tools import eval_run as tools_eval_run

from .tearsheet_html import generate_tearsheet


def _derive_parts(run_dir: Path) -> tuple[str, str, str]:
    """Extract symbol, frame and run_id from a run directory."""
    parts = run_dir.resolve().parts
    if len(parts) < 3:
        raise ValueError("run directory must be results/<symbol>/<frame>/<run>")
    symbol, frame, run_id = parts[-3], parts[-2], parts[-1]
    if run_id == "latest" and (run_dir.is_symlink() or run_dir.exists()):
        run_id = Path(run_dir).resolve().name
    return symbol, frame, run_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a run directory")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--tearsheet-html", action="store_true")
    args = p.parse_args(argv)

    symbol, frame, run_id = _derive_parts(args.run_dir)
    # delegate to tools.eval_run
    cmd = ["--symbol", symbol, "--frame", frame, "--run-id", run_id]
    if args.tearsheet_html:
        cmd.append("--tearsheet")
    rc = tools_eval_run.main(cmd)
    if args.tearsheet_html:
        generate_tearsheet(args.run_dir)
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
