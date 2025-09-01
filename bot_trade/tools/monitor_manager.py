import argparse
import sys
import time
from pathlib import Path

import logging

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


def export_charts_for_run(
    paths: RunPaths,
    wait_sec: int = 10,
    min_images: int = 1,
    debug: bool = False,
) -> tuple[Path, int, int, int]:
    """Export reward/step charts for ``paths``.

    Returns ``(charts_dir, image_count, rows_reward, rows_step)``.
    """

    from bot_trade.tools import export_charts as ec

    ensure_contract(paths.as_dict())

    reward_file = paths.results / "reward" / "reward.log"
    step_file = paths.logs / "step_log.csv"

    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if reward_file.exists() or step_file.exists():
            break
        time.sleep(0.5)

    charts_dir, count, rows_reward, rows_step = ec.export_run(paths, debug=debug)
    return charts_dir, count, rows_reward, rows_step


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
    ap.add_argument("--debug-export", action="store_true")
    args = ap.parse_args(argv)

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
    print(f"[CHARTS] dir={charts_path} images={count}", flush=True)
    return 0 if count > 0 else 2


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        code = 3
    sys.exit(code)
