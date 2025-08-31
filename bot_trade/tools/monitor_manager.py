import argparse
import sys
import time
import os
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import matplotlib

from bot_trade.config.rl_paths import RunPaths, get_root, ensure_contract


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _infer_run_id(symbol: str, frame: str, results_root: Path) -> Tuple[str | None, List[Path]]:
    checked: List[Path] = []
    base = results_root / symbol / frame
    if base.exists():
        run_dirs = sorted(
            [d for d in base.iterdir() if d.is_dir() and not d.is_symlink()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in run_dirs:
            reward = d / "reward" / "reward.log"
            checked.append(reward)
            if reward.exists() and reward.stat().st_size > 0:
                return d.name, checked
    # fallback to legacy logs structure
    logs_base = results_root.parent / "logs" / symbol / frame
    if logs_base.exists():
        run_dirs = sorted(
            [d for d in logs_base.iterdir() if d.is_dir() and not d.is_symlink()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in run_dirs:
            reward = d / "reward.log"
            checked.append(reward)
            if reward.exists() and reward.stat().st_size > 0:
                return d.name, checked
        for d in run_dirs:
            step = d / "step_log.csv"
            checked.append(step)
            if step.exists() and step.stat().st_size > 0:
                return d.name, checked
    snaps = results_root.parent / "memory" / "snapshots"
    checked.append(snaps)
    return None, checked


def _export_charts(rp: RunPaths, charts_dir: Path) -> Tuple[Path, int]:
    import matplotlib.pyplot as plt

    charts_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    def save_fig(fig, name: str) -> None:
        nonlocal count
        out = charts_dir / name
        tmp = out.with_suffix(out.suffix + ".tmp")
        fig.savefig(tmp, format="png")
        plt.close(fig)
        os.replace(tmp, out)
        count += 1

    reward_file = rp.results / "reward" / "reward.log"
    if not reward_file.exists():
        reward_file = rp.logs / "reward.log"
    step_file = rp.logs / "step_log.csv"
    risk_file = rp.logs / "risk_log.csv"

    try:
        df = pd.read_csv(reward_file)
        if not df.empty:
            fig, ax = plt.subplots()
            ax.plot(df.get("global_step", df.index), df.get("reward_total", df.get("reward", 0)))
            save_fig(fig, "reward_curve.png")
    except Exception:
        pass

    try:
        df = pd.read_csv(step_file)
        if not df.empty and "metric" in df.columns:
            for metric in list(df["metric"].unique())[:4]:
                m = df[df["metric"] == metric]
                if not m.empty:
                    fig, ax = plt.subplots()
                    ax.plot(m["step"], m["value"])
                    save_fig(fig, f"{metric}.png")
    except Exception:
        pass

    try:
        df = pd.read_csv(risk_file)
        if not df.empty and "reason" in df.columns:
            counts = df["reason"].value_counts().head(10)
            fig, ax = plt.subplots()
            counts.plot.bar(ax=ax)
            save_fig(fig, "risk_flags.png")
    except Exception:
        pass

    if count < 5:
        for i in range(count, 5):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "n/a", ha="center")
            ax.set_axis_off()
            save_fig(fig, f"extra_{i}.png")

    return charts_dir.resolve(), max(count, 5)


def export_charts_for_run(symbol: str, frame: str, run_id: str, base: str | None = None, headless: bool = True) -> Tuple[Path, int]:
    """Export charts for a given run and return the directory and image count."""
    root = Path(base) if base else get_root()
    rp = RunPaths(symbol, frame, run_id, root=root)
    ensure_contract(rp.as_dict())
    if headless:
        matplotlib.use("Agg", force=True)
    deadline = time.time() + 30
    reward_new = rp.results / "reward" / "reward.log"
    reward_old = rp.logs / "reward.log"
    step_file = rp.logs / "step_log.csv"
    while time.time() < deadline:
        if (reward_new.exists() and reward_new.stat().st_size > 0) or \
           (reward_old.exists() and reward_old.stat().st_size > 0) or \
           (step_file.exists() and step_file.stat().st_size > 0):
            break
        time.sleep(0.5)

    charts_dir = rp.reports / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    try:
        from bot_trade.tools.export_charts import export_for_run
        export_for_run(rp, charts_dir)
    except Exception:
        pass
    count = len(list(charts_dir.glob("*.png")))
    if count < 5:
        _, count = _export_charts(rp, charts_dir)
    return charts_dir.resolve(), count


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser("monitor manager")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--run-id")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)

    if args.headless:
        matplotlib.use("Agg")

    root = Path(args.base) if args.base else get_root()
    results_root = Path(args.data_dir) if args.data_dir else root / "results"

    run_id = args.run_id
    if run_id == "latest":
        base = results_root / args.symbol / args.frame
        try:
            run_dirs = sorted(
                [d for d in base.iterdir() if d.is_dir() and not d.is_symlink()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            run_id = run_dirs[0].name if run_dirs else None
        except Exception:
            run_id = None
    if not run_id:
        rid, checked = _infer_run_id(args.symbol, args.frame, results_root)
        if not rid:
            files = "\n".join(str(p) for p in checked)
            print(f"[ERROR] Could not determine run_id\nChecked:\n{files}", file=sys.stderr)
            return 2
        run_id = rid

    charts_path, count = export_charts_for_run(
        args.symbol, args.frame, run_id, base=str(root), headless=args.headless
    )
    abs_dir = charts_path.resolve()
    print(f"[CHARTS] dir={abs_dir} images={count}", flush=True)
    if count == 0:
        print("[ERROR] no charts generated", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        code = 3
    sys.exit(code)
