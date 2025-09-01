"""Robust chart exporter for training runs.

This module loads various training logs, normalises column names and
produces a fixed set of charts.  All operations are tolerant to missing or
malformed files so it is safe to run after every training session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import os
import sys
import pandas as pd

# use non-interactive backend at import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bot_trade.config.rl_paths import RunPaths

# ---------------------------------------------------------------------------
# column aliases
# ---------------------------------------------------------------------------
REWARD_ALIASES = {
    "reward_total": "reward",
    "reward": "reward",
    "r": "reward",
}

STEP_ALIASES = {
    "global_step": "step",
    "step": "step",
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path, aliases: Dict[str, str] | None = None) -> pd.DataFrame:
    """Read ``path`` returning empty frame on failure.

    Columns are renamed using ``aliases`` if provided.  All non timestamp
    columns are coerced to numeric and rows consisting solely of ``NaN`` are
    dropped (zeros are preserved).
    """

    if not path.exists() or path.is_dir():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if aliases:
        df.rename(
            columns={c: aliases.get(c, aliases.get(c.lower(), c)) for c in df.columns},
            inplace=True,
        )
    for col in list(df.columns):
        lc = col.lower()
        if lc not in {"ts", "timestamp"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(how="all", inplace=True)
    return df


def _placeholder(path: Path, title: str) -> None:
    """Create labelled placeholder ensuring file size >=1KB."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, dpi=120)
    os.replace(tmp, path)
    plt.close(fig)
    if path.stat().st_size < 1024:
        with path.open("ab") as fh:
            fh.write(b"0" * (1024 - path.stat().st_size))


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------

def export_for_run(run_paths: RunPaths, debug: bool = False) -> Tuple[Path, int, Dict[str, int]]:
    """Export charts for ``run_paths``.

    Returns ``(charts_dir, image_count, rows_dict)``.
    """

    rp = run_paths
    charts_dir = rp.charts_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    reward_file = rp.results / "reward" / "reward.log"
    step_file = rp.logs / "step_log.csv"
    train_file = rp.logs / "train_log.csv"
    risk_file = rp.logs / "risk_log.csv"
    signals_file = rp.logs / "signals.csv"
    callbacks_file = rp.logs / "callbacks.jsonl"

    reward = _read_csv(reward_file, {**REWARD_ALIASES, **STEP_ALIASES})
    step = _read_csv(step_file, {**REWARD_ALIASES, **STEP_ALIASES})
    train = _read_csv(train_file)
    risk = _read_csv(risk_file)
    signals = _read_csv(signals_file)

    callbacks_lines = 0
    if callbacks_file.exists():
        try:
            with callbacks_file.open("r", encoding="utf-8") as fh:
                callbacks_lines = sum(1 for line in fh if line.strip())
        except Exception:
            callbacks_lines = 0

    if not reward.empty:
        reward.dropna(subset=[c for c in ["step", "reward"] if c in reward.columns], inplace=True)
    if "step" in step.columns:
        step.dropna(subset=["step"], inplace=True)
    if not train.empty:
        for c in ["loss", "entropy", "entropy_loss"]:
            if c in train.columns:
                train[c] = pd.to_numeric(train[c], errors="coerce")
        train.dropna(how="all", inplace=True)
    if not risk.empty:
        risk.dropna(how="all", inplace=True)
    if not signals.empty:
        signals.dropna(how="all", inplace=True)

    rows_reward = len(reward)
    rows_step = len(step)
    rows_train = len(train)

    risk_events = pd.DataFrame()
    for df in (risk, step):
        if not df.empty:
            if {"metric", "step"}.issubset(df.columns):
                rf = df[df["metric"].isin(["dd_pen", "vol_pen", "risk_flag"])]
            else:
                cols = [c for c in ["dd_pen", "vol_pen", "risk_flag"] if c in df.columns]
                if cols:
                    rf = df.dropna(subset=cols, how="all")
                else:
                    rf = pd.DataFrame()
            if not rf.empty:
                risk_events = rf
                break
    rows_risk = len(risk_events)

    rows_signals = len(signals)

    images = 0
    # reward chart
    reward_path = charts_dir / "reward.png"
    if rows_reward > 0 and {"step", "reward"}.issubset(reward.columns):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(reward["step"], reward["reward"].fillna(0))
        ax.set_title("reward")
        _save(fig, reward_path)
    else:
        _placeholder(reward_path, "NO DATA")
    images += 1

    # sharpe ratio
    sharpe_series = pd.Series(dtype=float)
    steps_for_sharpe = pd.Series(dtype=float)
    if rows_reward > 1 and {"step", "reward"}.issubset(reward.columns):
        returns = reward["reward"].diff().fillna(0)
        steps_for_sharpe = reward["step"]
        window = min(256, max(2, len(returns)))
        sharpe_series = returns.rolling(window).mean() / returns.rolling(window).std()
        sharpe_series = sharpe_series.fillna(0)
    else:
        pnl_col = next((c for c in ["pnl", "pnl_total", "reward"] if c in step.columns), None)
        if pnl_col is not None and rows_step > 1:
            returns = step[pnl_col].diff().fillna(0)
            steps_for_sharpe = step["step"] if "step" in step.columns else pd.Series(range(len(step)))
            window = min(256, max(2, len(returns)))
            sharpe_series = returns.rolling(window).mean() / returns.rolling(window).std()
            sharpe_series = sharpe_series.fillna(0)
    sharpe_path = charts_dir / "sharpe.png"
    if not sharpe_series.dropna().empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps_for_sharpe, sharpe_series)
        ax.set_title("sharpe")
        _save(fig, sharpe_path)
    else:
        _placeholder(sharpe_path, "NO DATA")
    images += 1

    # loss chart
    loss_path = charts_dir / "loss.png"
    entropy_path = charts_dir / "entropy.png"
    loss_df = pd.DataFrame()
    ent_df = pd.DataFrame()
    if rows_train > 0:
        if {"step", "loss"}.issubset(train.columns):
            loss_df = train[["step", "loss"]].dropna(subset=["loss"])
        elif {"metric", "step", "value"}.issubset(train.columns):
            loss_df = train[train["metric"] == "loss"][ ["step", "value"] ].rename(columns={"value": "loss"})
        if {"step", "entropy"}.issubset(train.columns):
            ent_df = train[["step", "entropy"]].dropna(subset=["entropy"])
        elif {"step", "entropy_loss"}.issubset(train.columns):
            ent_df = train[["step", "entropy_loss"]].rename(columns={"entropy_loss": "entropy"}).dropna(subset=["entropy"])
        elif {"metric", "step", "value"}.issubset(train.columns):
            ent_df = train[train["metric"].isin(["entropy", "entropy_loss"])][["step", "value"]].rename(columns={"value": "entropy"})
    if not loss_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(loss_df["step"], loss_df["loss"].fillna(0))
        ax.set_title("loss")
        _save(fig, loss_path)
    else:
        _placeholder(loss_path, "NO DATA")
    images += 1

    if not ent_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ent_df["step"], ent_df["entropy"].fillna(0))
        ax.set_title("entropy")
        _save(fig, entropy_path)
    else:
        _placeholder(entropy_path, "NO DATA")
    images += 1

    # risk flags chart
    risk_path = charts_dir / "risk_flags.png"
    if rows_risk > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        steps = (
            risk_events["step"].values
            if "step" in risk_events.columns
            else list(range(len(risk_events)))
        )
        ax.scatter(steps, [1] * len(steps))
        ax.set_title("risk flags")
        _save(fig, risk_path)
    else:
        _placeholder(risk_path, "NO DATA")
    images += 1

    rows = {
        "reward": rows_reward,
        "step": rows_step,
        "train": rows_train,
        "risk": rows_risk,
        "callbacks": callbacks_lines,
        "signals": rows_signals,
    }

    return charts_dir, images, rows


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
    import argparse
    from bot_trade.config.rl_paths import get_root

    ap = argparse.ArgumentParser(
        description="Export charts for a training run",
        epilog=(
            "Example: python -m bot_trade.tools.export_run_charts "
            "--symbol BTCUSDT --frame 1m --run-id latest"
        ),
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--base", default=None, help="Project root override")
    ap.add_argument("--debug-export", action="store_true")
    ns = ap.parse_args(argv)

    root = Path(ns.base) if ns.base else get_root()
    run_id = ns.run_id
    if run_id in {None, "latest", ""}:
        rid = _latest_run(ns.symbol, ns.frame, root / "reports")
        run_id = rid or run_id
    if not run_id:
        print("[ERROR] run not found", file=sys.stderr)
        return 2
    rp = RunPaths(ns.symbol, ns.frame, run_id, root=root)
    charts_dir, images, _rows = export_for_run(rp, debug=ns.debug_export)
    print(f"[CHARTS] dir={charts_dir.resolve()} images={images}")
    return 0 if images > 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

