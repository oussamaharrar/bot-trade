"""Batch export of analytics charts and tables."""
from __future__ import annotations

# ``analytics_common`` pulls in ``matplotlib.pyplot`` during import. To ensure
# headless operation we must switch the backend *before* that happens. Always
# force the ``Agg`` backend so plots can be generated without a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  # backend already set

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from bot_trade.tools.analytics_common import (
    load_reward_df,
    load_trades_df,
    load_steps_df,
    load_decisions_df,
    compute_equity,
    compute_drawdown,
    compute_sharpe,
    penalties_breakdown,
    signals_agg,
    plot_line,
    plot_area,
    plot_bar,
    table_to_image,
    git_short_hash,
    wait_for_first_write,
)
from bot_trade.tools.paths import results_dir, report_dir

import pandas as pd


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

REWARD_ALIASES = ["reward", "reward_total"]
STEP_ALIASES = {"step": ["global_step", "step"], "value": ["reward", "reward_total", "value"]}

# ``reward.log`` and step logs may expose a variety of column names. Normalising
# to canonical ``reward``/``step`` keeps downstream code simple.
COL_ALIASES = {
    "reward_total": "reward",
    "reward": "reward",
    "r": "reward",
    "global_step": "step",
    "step": "step",
    "steps": "step",
    "t": "step",
}

CHARTS = [
    "reward",
    "equity",
    "drawdown",
    "sharpe",
    "penalties",
    "trades",
    "signals",
    "tables",
    "exposure",
    "fees",
    "positions",
]


def export(base: str, symbol: str, frame: str, out_dir: str, charts, limit, rollwin, svg: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    reward = load_reward_df(base, symbol, frame, limit).rename(columns=COL_ALIASES)
    steps = load_steps_df(base, symbol, frame, limit).rename(columns=COL_ALIASES)
    trades = load_trades_df(base, symbol, frame, limit)
    decisions = load_decisions_df(base, symbol, frame, limit)
    eq = compute_equity(trades)
    dd = compute_drawdown(eq)
    pen = penalties_breakdown(steps)
    sig = signals_agg(decisions)
    meta = {"git": git_short_hash(), "generated": datetime.utcnow().isoformat()}

    def maybe_svg(path_png):
        if not svg:
            return
        base, _ = os.path.splitext(path_png)
        return base + '.svg'

    # ensure numeric inputs; drop non-numeric columns/rows
    for df in (reward, steps):
        for col in list(df.columns):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(axis=1, how="all", inplace=True)
        df.dropna(inplace=True)

    reward_series = (
        reward["reward"]
        if "reward" in reward.columns
        else reward.iloc[:, 0]
        if not reward.empty
        else reward
    )

    if "reward" in charts and not reward_series.empty:
        png = os.path.join(out_dir, "reward.png")
        plot_line(reward_series.rolling(rollwin).mean(), png, "reward")
        if svg:
            plot_line(reward_series.rolling(rollwin).mean(), maybe_svg(png), "reward")
    if "equity" in charts:
        png = os.path.join(out_dir,'equity.png')
        plot_line(eq, png, 'equity')
        if svg: plot_line(eq, maybe_svg(png), 'equity')
    if "drawdown" in charts:
        png = os.path.join(out_dir,'drawdown.png')
        plot_line(dd, png, 'drawdown')
        if svg: plot_line(dd, maybe_svg(png), 'drawdown')
    if "sharpe" in charts:
        s = compute_sharpe(eq)
        png = os.path.join(out_dir,'sharpe.png')
        table_to_image(pd.DataFrame({'sharpe':[s]}), png)
        if svg: table_to_image(pd.DataFrame({'sharpe':[s]}), maybe_svg(png))
    if "penalties" in charts:
        png = os.path.join(out_dir,'penalties.png')
        plot_area(pen, png, 'penalties')
        if svg: plot_area(pen, maybe_svg(png), 'penalties')
    if "signals" in charts:
        png = os.path.join(out_dir,'signals_top.png')
        plot_bar(sig.set_index('signal'), png, 'signals')
        if svg: plot_bar(sig.set_index('signal'), maybe_svg(png), 'signals')
    if "tables" in charts:
        png = os.path.join(out_dir,'kpis_table.png')
        table_to_image(pd.DataFrame({'kpi':['sharpe','maxdd'], 'value':[compute_sharpe(eq), dd.max() if not dd.empty else 0]}), png)
        if svg: table_to_image(pd.DataFrame({'kpi':['sharpe','maxdd'], 'value':[compute_sharpe(eq), dd.max() if not dd.empty else 0]}), maybe_svg(png))
    if "exposure" in charts and not trades.empty and 'qty' in trades.columns:
        png = os.path.join(out_dir,'exposure.png')
        plot_line(trades['qty'].abs().cumsum(), png, 'exposure')
        if svg: plot_line(trades['qty'].abs().cumsum(), maybe_svg(png), 'exposure')
    if "fees" in charts and not trades.empty and 'fee' in trades.columns:
        png = os.path.join(out_dir,'fees.png')
        plot_line(trades['fee'].cumsum(), png, 'fees')
        if svg: plot_line(trades['fee'].cumsum(), maybe_svg(png), 'fees')
    if "positions" in charts and not trades.empty and 'qty' in trades.columns:
        png = os.path.join(out_dir,'positions_timeline.png')
        plot_line(trades['qty'].cumsum(), png, 'positions')
        if svg: plot_line(trades['qty'].cumsum(), maybe_svg(png), 'positions')
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    pngs = [
        p
        for p in Path(out_dir).glob("*.png")
        if p.is_file() and not p.is_symlink() and p.stat().st_size > 1024
    ]
    count = len(pngs)
    logging.info("[CHARTS] dir=%s images=%d", Path(out_dir).resolve(), count)
    return count


# ---------------------------------------------------------------------------
# Minimal exporter for RunPaths (reward/step charts only)
# ---------------------------------------------------------------------------

from bot_trade.config.rl_paths import RunPaths


def export_run(paths: RunPaths, debug: bool = False) -> tuple[Path, int, int, int]:
    """Export reward/step charts for a completed run.

    Returns ``(charts_dir, image_count, rows_reward, rows_step)``.
    """

    reward_file = paths.results / "reward" / "reward.log"
    step_file = paths.logs / "step_log.csv"
    charts_dir = paths.reports / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Load reward log
    # --------------------
    if reward_file.is_file():
        df_reward = pd.read_csv(reward_file, encoding="utf-8", low_memory=False)
        for col in REWARD_ALIASES:
            if col in df_reward.columns:
                df_reward.rename(columns={col: "reward"}, inplace=True)
                break
        df_reward["reward"] = pd.to_numeric(df_reward.get("reward"), errors="coerce")
    else:
        df_reward = pd.DataFrame()
    rows_reward = int(df_reward.get("reward").notna().sum()) if "reward" in df_reward else 0

    # --------------------
    # Load step log
    # --------------------
    if step_file.is_file():
        df_step = pd.read_csv(step_file, encoding="utf-8", low_memory=False)
        for canon, aliases in STEP_ALIASES.items():
            for a in aliases:
                if a in df_step.columns:
                    df_step.rename(columns={a: canon}, inplace=True)
                    break
        for col in list(df_step.columns):
            df_step[col] = pd.to_numeric(df_step[col], errors="coerce")
    else:
        df_step = pd.DataFrame()
    rows_step = int(df_step.notna().any(axis=1).sum()) if not df_step.empty else 0

    if debug:
        print(f"[DEBUG_EXPORT] reward_file={reward_file} step_file={step_file}")
        print(f"[DEBUG_EXPORT] reward_rows={rows_reward} step_rows={rows_step}")

    # --------------------
    # Plotting
    # --------------------
    def _save(fig, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    if rows_reward >= 1:
        fig, ax = plt.subplots()
        x = df_reward["global_step"].values if "global_step" in df_reward.columns else df_reward.index
        ax.plot(x, df_reward["reward"].fillna(0).values)
        ax.set_title(f"Reward (rows={rows_reward})")
        _save(fig, charts_dir / "reward.png")
    elif rows_step >= 1:
        fig, ax = plt.subplots()
        if "step" in df_step.columns:
            ax.plot(df_step["step"].values)
        else:
            ax.plot(range(rows_step))
        ax.set_title(f"Steps (rows={rows_step})")
        _save(fig, charts_dir / "steps.png")
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "NO DATA", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, charts_dir / "empty.png")

    pngs = [
        p
        for p in charts_dir.glob("*.png")
        if p.is_file() and not p.is_symlink() and p.stat().st_size > 1024
    ]
    count = len(pngs)
    return charts_dir.resolve(), count, rows_reward, rows_step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--debug-export", action="store_true")
    args = ap.parse_args()

    from bot_trade.config.rl_paths import get_root

    root = Path(args.base) if args.base else get_root()
    rp = RunPaths(args.symbol, args.frame, args.run_id, root=root)
    _, count, _, _ = export_run(rp, debug=args.debug_export)
    return 0 if count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
