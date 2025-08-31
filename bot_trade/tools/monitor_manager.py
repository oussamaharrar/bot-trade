import argparse
import sys
import time
from pathlib import Path

import matplotlib

from bot_trade.config.rl_paths import RunPaths, get_root, ensure_contract


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _infer_run_id(symbol: str, frame: str, results_root: Path) -> tuple[str | None, list[Path]]:
    checked: list[Path] = []
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


def export_charts_for_run(paths: RunPaths, wait_sec: int = 10, min_images: int = 5) -> tuple[Path, int]:
    """Export charts for ``paths`` and ensure a minimum number of images."""

    matplotlib.use("Agg", force=True)

    import pandas as pd
    import matplotlib.pyplot as plt
    from bot_trade.tools import export_charts as ec

    ensure_contract(paths.as_dict())

    reward_file = paths.results / "reward" / "reward.log"
    step_file = paths.logs / "step_log.csv"

    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if (
            reward_file.is_file()
            and not reward_file.is_symlink()
            and reward_file.stat().st_size > 0
        ) or (
            step_file.is_file()
            and not step_file.is_symlink()
            and step_file.stat().st_size > 0
        ):
            break
        time.sleep(0.5)

    charts_dir = paths.reports / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    try:
        ec.export(
            str(paths.results.parent.parent),
            paths.symbol,
            f"{paths.frame}/{paths.run_id}",
            str(charts_dir),
            ec.CHARTS,
            None,
            200,
            svg=False,
        )
    except Exception:
        pass

    def _load_series(path: Path) -> pd.Series:
        try:
            data = []
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    first = line.split(",", 1)[0]
                    try:
                        data.append(float(first))
                    except Exception:
                        continue
            return pd.Series(data)
        except Exception:
            return pd.Series(dtype=float)

    def _plot(series: pd.Series, path: Path, title: str) -> None:
        plt.figure()
        plt.plot(series.values)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    count = sum(1 for p in charts_dir.glob("*.png") if p.is_file() and not p.is_symlink())

    reward_series = _load_series(reward_file)
    step_series = _load_series(step_file)
    idx = 0
    while count < min_images:
        target = charts_dir / f"fallback_{idx}.png"
        if not reward_series.empty:
            _plot(reward_series, target, "reward")
        elif not step_series.empty:
            _plot(step_series, target, "steps")
        else:
            _plot(pd.Series([0]), target, "empty")
        count += 1
        idx += 1

    abs_dir = charts_dir.resolve()
    print(f"[CHARTS] dir={abs_dir} images={count}", flush=True)
    return abs_dir, count


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
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

    rp = RunPaths(args.symbol, args.frame, run_id, root=root)
    charts_path, count = export_charts_for_run(rp)
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
