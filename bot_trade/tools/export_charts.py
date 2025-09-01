"""Minimal multi-source exporter for training runs.

This module aggregates log CSVs produced during training and generates a
set of charts/summary files for a single run.  The behaviour is purposely
conservative so it can run in limited development environments.  Missing
inputs result in placeholder images so downstream tooling never fails.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# matplotlib must be configured *before* importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bot_trade.config.rl_paths import RunPaths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALIAS_REWARD = {"reward_total": "reward", "global_step": "step"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except Exception:
        return pd.DataFrame()


def _placeholder(path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_for_run(run_paths: RunPaths, debug: bool = False) -> Dict[str, Any]:
    """Build charts and summaries for ``run_paths``.

    Returns a dictionary with the chart directory, number of images and row
    counts for each ingested source.
    """

    charts_dir = run_paths.charts_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    reward_file = run_paths.results / "reward" / "reward.log"
    step_file = run_paths.logs / "step_log.csv"
    train_file = run_paths.logs / "train_log.csv"
    risk_file = run_paths.logs / "risk_log.csv"
    callbacks_file = run_paths.logs / "callbacks_log.csv"
    signals_dir = run_paths.logs / "signals_log"

    # --------------------
    # Load data
    # --------------------
    reward = _read_csv(reward_file).rename(columns=ALIAS_REWARD)
    step = _read_csv(step_file)
    train = _read_csv(train_file)
    risk = _read_csv(risk_file)
    callbacks = _read_csv(callbacks_file)

    signals_rows = 0
    if signals_dir.exists():
        for fp in signals_dir.glob("*.csv"):
            df = _read_csv(fp)
            signals_rows += len(df)

    # numeric coercion
    for df in (reward, step, train, risk, callbacks):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows where step is NaN for reward/step/train
    for df in (reward, step, train):
        if "step" in df.columns:
            df.dropna(subset=["step"], inplace=True)

    rows_reward = len(reward)
    rows_step = len(step)
    rows_train = len(train)
    rows_risk = len(risk)
    rows_callbacks = len(callbacks)

    # --------------------
    # Charts
    # --------------------
    images = 0

    def save_fig(fig: plt.Figure, name: str) -> None:
        nonlocal images
        path = charts_dir / name
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        images += 1

    if rows_reward > 0 and "step" in reward and "reward" in reward:
        fig, ax = plt.subplots()
        ax.plot(reward["step"], reward["reward"].fillna(0))
        ax.set_title("reward vs step")
        save_fig(fig, "reward.png")

        win = max(1, min(64, len(reward) // 4))
        fig, ax = plt.subplots()
        ax.plot(reward["step"], reward["reward"].rolling(win).mean())
        ax.set_title("reward rolling mean")
        save_fig(fig, "reward_ma.png")
    else:
        _placeholder(charts_dir / "reward.png", "NO REWARD DATA")
        _placeholder(charts_dir / "reward_ma.png", "NO REWARD DATA")
        images += 2

    if rows_train > 0 and {"step", "metric", "value"}.issubset(train.columns):
        loss = train[train["metric"] == "loss"]
        if len(loss) > 0:
            fig, ax = plt.subplots()
            ax.plot(loss["step"], loss["value"].fillna(0))
            ax.set_title("loss")
            save_fig(fig, "loss.png")
        else:
            _placeholder(charts_dir / "loss.png", "NO LOSS DATA")
            images += 1

        ent = train[train["metric"] == "entropy"]
        if len(ent) > 0:
            fig, ax = plt.subplots()
            ax.plot(ent["step"], ent["value"].fillna(0))
            ax.set_title("entropy")
            save_fig(fig, "entropy.png")
        else:
            _placeholder(charts_dir / "entropy.png", "NO ENTROPY DATA")
            images += 1
    else:
        _placeholder(charts_dir / "loss.png", "NO LOSS DATA")
        _placeholder(charts_dir / "entropy.png", "NO ENTROPY DATA")
        images += 2

    if rows_risk > 0:
        fig, ax = plt.subplots()
        ax.bar(["risk"], [rows_risk])
        ax.set_title("risk flags")
        save_fig(fig, "risk_flags.png")
    elif signals_rows > 0:
        fig, ax = plt.subplots()
        ax.bar(["signals"], [signals_rows])
        ax.set_title("risk flags")
        save_fig(fig, "risk_flags.png")
    else:
        _placeholder(charts_dir / "risk_flags.png", "NO RISK DATA")
        images += 1

    # Sharpe ratio from reward
    if rows_reward > 1 and reward["reward"].std() > 0:
        sharpe = reward["reward"].mean() / reward["reward"].std() * (len(reward) ** 0.5)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Sharpe: {sharpe:.3f}", ha="center", va="center")
        ax.set_axis_off()
        save_fig(fig, "sharpe.png")
    else:
        _placeholder(charts_dir / "sharpe.png", "NO SHARPE")
        images += 1

    # Ensure at least 5 images
    existing = [p for p in charts_dir.glob("*.png") if p.is_file() and not p.is_symlink()]
    while len(existing) < 5:
        name = f"fallback_{len(existing)+1}.png"
        _placeholder(charts_dir / name, "NO DATA")
        existing = [p for p in charts_dir.glob("*.png") if p.is_file() and not p.is_symlink()]
    images = len(existing)

    # --------------------
    # Summary files
    # --------------------
    rows = {
        "reward": rows_reward,
        "step": rows_step,
        "train": rows_train,
        "risk": rows_risk,
        "callbacks": rows_callbacks,
        "signals": signals_rows,
    }
    summary_csv = run_paths.summary_csv_path
    summary_json = run_paths.summary_json_path
    df_summary = pd.DataFrame([
        {
            "rows_reward": rows_reward,
            "rows_step": rows_step,
            "rows_train": rows_train,
            "rows_risk": rows_risk,
            "rows_callbacks": rows_callbacks,
            "rows_signals_total": signals_rows,
            "images": images,
        }
    ])
    _atomic_write(summary_csv, df_summary.to_csv(index=False))
    summary_payload = {
        "rows_reward": rows_reward,
        "rows_step": rows_step,
        "rows_train": rows_train,
        "rows_risk": rows_risk,
        "rows_callbacks": rows_callbacks,
        "rows_signals_total": signals_rows,
        "images": images,
        "charts_dir": str(charts_dir.resolve()),
    }
    _atomic_write(summary_json, json.dumps(summary_payload, ensure_ascii=False, indent=2))

    if debug:
        print(
            "[DEBUG_EXPORT] reward_rows=%d step_rows=%d train_rows=%d risk_rows=%d callbacks_rows=%d signals_rows=%d"
            % (
                rows_reward,
                rows_step,
                rows_train,
                rows_risk,
                rows_callbacks,
                signals_rows,
            )
        )

    return {
        "charts_dir": str(charts_dir.resolve()),
        "images": images,
        "rows": rows,
    }


# Backwards compatibility ----------------------------------------------------


def export_run(paths: RunPaths, debug: bool = False):  # pragma: no cover - shim
    res = export_for_run(paths, debug=debug)
    return Path(res["charts_dir"]), res["images"], res["rows"]["reward"], res["rows"]["step"]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI helper
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--debug-export", action="store_true")
    ns = ap.parse_args(argv)

    from bot_trade.config.rl_paths import get_root

    root = Path(ns.base) if ns.base else get_root()
    rp = RunPaths(ns.symbol, ns.frame, ns.run_id, root=root)
    info = export_for_run(rp, debug=ns.debug_export)
    return 0 if info["images"] > 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

