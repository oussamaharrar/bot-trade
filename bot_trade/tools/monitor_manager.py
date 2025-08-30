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
    try:
        from bot_trade.config.rl_paths import get_root  # lazy

        return Path(get_root())
    except Exception:
        pass
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
        for col in ("run_id", "runId", "session_id", "session", "id", "run"):
            if col in header:
                idx = header.index(col)
                return last[idx] or None
    except Exception:
        return None
    return None


def auto_run_id(symbol: str, frame: str, root: Path) -> tuple[Optional[str], List[Path]]:
    checked: List[Path] = []
    mem_file = root / "memory" / "memory.json"
    checked.append(mem_file)
    if mem_file.exists():
        try:
            data = json.loads(mem_file.read_text(encoding="utf-8"))
            rid = data.get("last_run_id")
            if rid:
                return rid, checked
        except Exception:
            pass
    res = root / "results" / symbol / frame
    candidates = [
        res / "train_log.csv",
        res / "step_log.csv",
        res / "deep_rl_evaluation.csv",
    ]
    for c in candidates:
        checked.append(c)
        rid = read_run_id_from_csv(c)
        if rid:
            return rid, checked
    return None, checked


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Interactive monitor manager")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--refresh", type=int, default=10)
    ap.add_argument("--images-out")
    ap.add_argument("--base", default=None, help="Project root")
    ap.add_argument("--run-id", help="Run identifier")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--headless", action="store_true")
    return ap


def misuse_check() -> bool:
    msg = (
        "Please run from the project root:\n"
        "  python -m bot_trade.tools.monitor_manager [flags]\n"
        "or use the console script:\n"
        "  bot-monitor [flags]"
    )
    if __package__ is None and ("/" in sys.argv[0] or sys.argv[0].endswith(".py") or "\\" in sys.argv[0]):
        print(msg, file=sys.stderr)
        return True
    cwd = Path.cwd()
    if cwd.name == "bot_trade" and (cwd.parent / "pyproject.toml").exists():
        print(msg, file=sys.stderr)
        return True
    return False


def main(argv: List[str] | None = None) -> None:
    if misuse_check():
        raise SystemExit(2)

    parser = build_parser()
    args = parser.parse_args(argv)
    headless = args.no_wait or args.headless

    root = Path(args.base) if args.base else discover_root()
    res_dir = root / "results" / args.symbol / args.frame
    base_dir = str(res_dir)

    if not args.run_id:
        rid, checked = auto_run_id(args.symbol, args.frame, root)
        if rid:
            print(f"Using run_id={rid}", flush=True)
            args.run_id = rid
        else:
            files = "\n".join(f"- {p}" for p in checked)
            print(
                "[ERROR] Could not determine run_id. Checked:\n" + files +
                f"\nExample: python -m bot_trade.tools.monitor_manager --symbol {args.symbol} --frame {args.frame} --run-id <id>",
                file=sys.stderr,
            )
            raise SystemExit(2)

    if headless:
        from bot_trade.tools.analytics_common import wait_for_first_write
        if not args.no_wait:
            print("Waiting for first log write...", flush=True)
            wait_for_first_write(base_dir, args.symbol, args.frame)
        logs_dir = res_dir / "logs"
        report_dir = root / "reports" / args.symbol / args.frame
        report_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import re, collections

        def _load_reward(path: Path):
            if not path.exists():
                return None
            try:
                df = pd.read_csv(path)
                if df.empty:
                    return None
                df = df.rename(columns={"global_step": "step", "reward_total": "reward"})
                return df
            except Exception:
                return None

        reward_df = _load_reward(logs_dir / "reward.log")
        try:
            train_df = pd.read_csv(logs_dir / "train_log.csv")
        except Exception:
            train_df = None
        try:
            step_df = pd.read_csv(logs_dir / "step_log.csv")
        except Exception:
            step_df = None
        try:
            risk_lines = (logs_dir / "risk.log").read_text(encoding="utf-8").splitlines()
        except Exception:
            risk_lines = []

        try:
            if reward_df is not None and not reward_df.empty:
                fig, ax1 = plt.subplots()
                ax1.plot(reward_df["step"], reward_df["reward"], label="reward")
                if "pnl" in reward_df.columns:
                    ax2 = ax1.twinx()
                    ax2.plot(reward_df["step"], reward_df["pnl"], color="orange", label="pnl")
                fig.savefig(report_dir / "reward.png", bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass

        try:
            if step_df is not None and not step_df.empty:
                pivot = step_df.pivot_table(index="step", columns="metric", values="value", aggfunc="last")
                cols = [c for c in ["value_loss", "policy_gradient_loss", "approx_kl", "learning_rate"] if c in pivot.columns]
                if cols:
                    fig, ax = plt.subplots()
                    pivot[cols].plot(ax=ax)
                    fig.savefig(report_dir / "loss.png", bbox_inches="tight")
                    plt.close(fig)
                if "explained_variance" in pivot.columns:
                    fig, ax = plt.subplots()
                    ax.plot(pivot.index, pivot["explained_variance"])
                    fig.savefig(report_dir / "variance.png", bbox_inches="tight")
                    plt.close(fig)
                if (step_df["metric"] == "action").any():
                    acts = step_df[step_df["metric"] == "action"]["value"].value_counts().sort_index()
                    fig, ax = plt.subplots()
                    acts.plot(kind="bar", ax=ax)
                    fig.savefig(report_dir / "actions.png", bbox_inches="tight")
                    plt.close(fig)
        except Exception:
            pass

        try:
            if risk_lines:
                counts = collections.Counter()
                for line in risk_lines:
                    m = re.search(r"flag=([A-Za-z0-9_]+)", line)
                    if m:
                        counts[m.group(1)] += 1
                if counts:
                    fig, ax = plt.subplots()
                    items = sorted(counts.items())
                    ax.bar([k for k, _ in items], [v for _, v in items])
                    fig.savefig(report_dir / "risk_flags.png", bbox_inches="tight")
                    plt.close(fig)
        except Exception:
            pass

        print("Headless refresh done.")
        sys.exit(0)

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
    raise SystemExit(main())

