import argparse
import sys
import time
from pathlib import Path

import matplotlib
import logging

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot_trade.config.rl_paths import RunPaths, get_root, ensure_contract

COL_ALIASES = {
    "reward_total": "reward",
    "reward": "reward",
    "r": "reward",
    "global_step": "step",
    "step": "step",
    "steps": "step",
    "t": "step",
}


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


def export_charts_for_run(
    paths: RunPaths,
    wait_sec: int = 10,
    min_images: int = 5,
    debug: bool = False,
) -> tuple[Path, int, int, int]:
    """Export charts and ensure a minimum number of non-empty images.

    Returns a tuple ``(charts_dir, image_count, rows_reward, rows_step)``.
    """

    import pandas as pd
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

    # ------------------------------------------------------------------
    # Load data with aliasing and numeric coercion
    # ------------------------------------------------------------------
    def _load_reward(path: Path) -> tuple[pd.Series, int]:
        data: list[float] = []
        raw = 0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    raw += 1
                    first = line.split(",", 1)[0]
                    try:
                        data.append(float(first))
                    except Exception:
                        continue
        except Exception:
            pass
        return pd.Series(data, dtype=float), raw

    def _load_steps(path: Path) -> tuple[pd.DataFrame, int]:
        raw = 0
        try:
            df = pd.read_csv(path)
            raw = len(df)
        except Exception:
            df = pd.DataFrame()
        df.rename(columns=COL_ALIASES, inplace=True)
        for col in list(df.columns):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(axis=1, how="all", inplace=True)
        df.dropna(inplace=True)
        return df, raw

    reward_series, reward_raw = _load_reward(reward_file)
    steps_df, steps_raw = _load_steps(step_file)
    rows_reward = int(len(reward_series))
    rows_step = int(len(steps_df))
    min_step = int(steps_df["step"].min()) if not steps_df.empty and "step" in steps_df else 0
    max_step = int(steps_df["step"].max()) if not steps_df.empty and "step" in steps_df else 0
    kept_pct = (rows_step / steps_raw * 100.0) if steps_raw else 0.0

    logging.info(
        "[EXPORT] run_id=%s rows_reward=%d rows_step=%d steps=[%s..%s] kept=%.1f%%",
        paths.run_id,
        rows_reward,
        rows_step,
        min_step,
        max_step,
        kept_pct,
    )

    if debug:
        print(
            f"[DEBUG_EXPORT] reward_file={reward_file} step_file={step_file}",
            flush=True,
        )
        print(
            f"[DEBUG_EXPORT] reward_rows={rows_reward} step_rows={rows_step}",
            flush=True,
        )
        if rows_step:
            sample = steps_df["step"].astype(float)
            print(
                f"[DEBUG_EXPORT] steps_sample={sample.head(3).tolist()}...{sample.tail(3).tolist()}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Run full exporter
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helper for plots / placeholders
    # ------------------------------------------------------------------
    def _plot(series: pd.Series, path: Path, title: str) -> None:
        fig, ax = plt.subplots()
        if series.dropna().nunique() > 1:
            ax.plot(series.values)
        if not ax.has_data() or series.dropna().nunique() < 2:
            ax.text(
                0.5,
                0.5,
                f"NO DATA (run={paths.run_id}) rows_reward={rows_reward} rows_step={rows_step} {path.name}",
                ha="center",
                va="center",
                wrap=True,
            )
            ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    # Replace empty images with placeholders
    for p in charts_dir.glob("*.png"):
        if not p.is_file() or p.is_symlink() or p.stat().st_size <= 1024:
            _plot(pd.Series(dtype=float), p, p.stem)

    pngs = [
        p
        for p in charts_dir.glob("*.png")
        if p.is_file() and not p.is_symlink() and p.stat().st_size > 1024
    ]
    count = len(pngs)

    # Ensure minimum image count
    idx = 0
    while count < min_images:
        target = charts_dir / f"fallback_{idx}.png"
        if rows_reward > 0:
            _plot(reward_series, target, "reward")
        elif rows_step > 0 and "step" in steps_df:
            _plot(steps_df["step"], target, "steps")
        else:
            _plot(pd.Series(dtype=float), target, "empty")
        if target.is_file() and target.stat().st_size > 1024:
            count += 1
        idx += 1

    abs_dir = charts_dir.resolve()
    print(f"[CHARTS] dir={abs_dir} images={count}", flush=True)
    return abs_dir, count, rows_reward, rows_step


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
    ap.add_argument("--debug-export", action="store_true")
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
    charts_path, count, _, _ = export_charts_for_run(rp, debug=args.debug_export)
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
