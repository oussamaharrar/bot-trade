"""Interactive monitor manager for training sessions."""
from __future__ import annotations

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from bot_trade.config.rl_paths import memory_dir


MENU = """
=== MONITOR MANAGER (symbol={symbol} frame={frame}) ===
[t] Live Ticker ({refresh}s)
[c] CLI Console
[a] Anomaly Watch ({refresh}s)
[r] Resource Monitor (1s)
[e] Batch Exporter (one-shot)
[q] Quit Manager
Choose: """


def discover_root(start: Path | None = None) -> Path:
    env = os.environ.get("BOT_TRADE_ROOT")
    if env:
        return Path(env).resolve()
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start


def read_run_id_from_csv(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return None
        header, *data = rows
        if not data:
            return None
        last = data[-1]
        for col in ("run_id", "runId", "session_id"):
            if col in header:
                idx = header.index(col)
                return last[idx] or None
    except Exception:
        return None
    return None


def auto_run_id(symbol: str, frame: str, root: Path) -> Optional[str]:
    res = root / "results" / symbol / frame / "logs"
    candidates = [
        res / "train_log.csv",
        res / "step_log.csv",
        res / "deep_rl_evaluation.csv",
    ]
    for c in candidates:
        rid = read_run_id_from_csv(c)
        if rid:
            return rid
    mem_file = memory_dir() / "memory.json"
    if mem_file.exists():
        try:
            data = json.loads(mem_file.read_text(encoding="utf-8"))
            return data.get("last_run_id")
        except Exception:
            return None
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Interactive monitor manager")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--refresh", type=int, default=10)
    ap.add_argument("--images-out")
    ap.add_argument("--base", default=None, help="Project root")
    ap.add_argument("--run-id", help="Run identifier")
    ap.add_argument("--no-wait", action="store_true")
    return ap


def misuse_check() -> bool:
    if __package__ is None and ("/" in sys.argv[0] or sys.argv[0].endswith(".py") or "\\" in sys.argv[0]):
        print(
            "You ran this with a file path. Use either:\n"
            "  python -m bot_trade.tools.monitor_manager [flags]\n"
            "  bot-monitor [flags]",
            file=sys.stderr,
        )
        return True
    cwd = Path.cwd()
    if cwd.name == "bot_trade" and (cwd.parent / "pyproject.toml").exists():
        print(
            "You are running from inside the package directory. cd to the project root before -m.",
            file=sys.stderr,
        )
        return True
    return False


def main(argv: List[str] | None = None) -> None:
    if misuse_check():
        raise SystemExit(2)

    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(args.base) if args.base else discover_root()
    res_dir = root / "results" / args.symbol / args.frame
    base_dir = str(res_dir)

    if not args.run_id:
        rid = auto_run_id(args.symbol, args.frame, root)
        if rid:
            print(f"Using run_id={rid}", flush=True)
            args.run_id = rid
        else:
            print(
                "[ERROR] Could not determine run_id. Checked train_log.csv, step_log.csv, deep_rl_evaluation.csv and memory.json.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    from bot_trade.tools.analytics_common import wait_for_first_write
    from bot_trade.tools.monitor_launch import launch_new_console

    img_dir = args.images_out.format(symbol=args.symbol, frame=args.frame) if args.images_out else None

    if not args.no_wait:
        print("Waiting for first log write...", flush=True)
        wait_for_first_write(base_dir, args.symbol, args.frame)

    while True:
        try:
            choice = input(
                MENU.format(symbol=args.symbol, frame=args.frame, refresh=args.refresh)
            ).strip().lower()[:1]
        except EOFError:
            break
        if choice == "q":
            break
        elif choice == "t":
            args_list = [
                "--symbol",
                args.symbol,
                "--frame",
                args.frame,
                "--refresh",
                str(args.refresh),
                "--base",
                base_dir,
                "--run-id",
                args.run_id,
            ]
            if img_dir:
                args_list += ["--images-out", img_dir]
            launch_new_console("LIVE-TICKER", "tools.live_ticker", args_list)
        elif choice == "c":
            launch_new_console(
                "CLI-CONSOLE",
                "tools.cli_console",
                ["--symbol", args.symbol, "--frame", args.frame, "--base", base_dir],
            )
        elif choice == "a":
            launch_new_console(
                "ANOMALY-WATCH",
                "tools.anomaly_watch",
                [
                    "--symbol",
                    args.symbol,
                    "--frame",
                    args.frame,
                    "--refresh",
                    str(args.refresh),
                    "--base",
                    base_dir,
                    "--run-id",
                    args.run_id,
                ],
            )
        elif choice == "r":
            launch_new_console(
                "RESOURCE-MON",
                "tools.resource_monitor",
                ["--symbol", args.symbol, "--frame", args.frame, "--base", base_dir],
            )
        elif choice == "e":
            out = img_dir or str(root / "report" / args.symbol / args.frame / args.run_id)
            launch_new_console(
                "BATCH-EXPORT",
                "tools.export_charts",
                [
                    "--symbol",
                    args.symbol,
                    "--frame",
                    args.frame,
                    "--base",
                    base_dir,
                    "--run-id",
                    args.run_id,
                    "--out",
                    out,
                ],
            )
            print(f"Exporting to {out}", flush=True)
        else:
            print("unknown choice", flush=True)


if __name__ == "__main__":
    main()

