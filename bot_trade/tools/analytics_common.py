"""Common analytics helpers for bot-trade tools."""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def read_csv_safely(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read CSV tolerating absence/growth."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        if limit and len(df) > limit:
            df = df.tail(limit)
        return df
    except Exception:
        return pd.DataFrame()


def read_jsonl_safely(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        df = pd.DataFrame(rows)
        if limit and len(df) > limit:
            df = df.tail(limit)
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def artifact_paths(base: str, symbol: str, frame: str) -> List[str]:
    """Return candidate artifact paths that indicate training activity."""
    root = os.path.join(base, symbol, frame)
    return [
        os.path.join(root, "train_log.csv"),
        os.path.join(root, "step_log.csv"),
        os.path.join(root, "evaluation.csv"),
        os.path.join(root, "logs", "entry_decisions.jsonl"),
    ]


def wait_for_first_write(
    base: str,
    symbol: str,
    frame: str,
    timeout: Optional[float] = None,
    poll: float = 0.5,
) -> Optional[str]:
    """Block until any artifact is written (size>0)."""
    paths = artifact_paths(base, symbol, frame)
    start = time.monotonic()
    while True:
        for p in paths:
            try:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    return p
            except Exception:
                pass
        if timeout is not None and (time.monotonic() - start) > timeout:
            return None
        time.sleep(poll)

# ---------------------------------------------------------------------------
# Column aliasing utilities
# ---------------------------------------------------------------------------

def find_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    for a in aliases:
        if a in df.columns:
            return a
    return None

REWARD_ALIASES = ["avg_reward", "ep_rew_mean", "mean_reward", "episode_reward", "reward"]
EQUITY_ALIASES = ["equity", "pnl", "cum_pnl", "balance"]
PENALTY_ALIASES = ["penalty", "risk_penalty", "slippage_penalty", "time_penalty", "vol_penalty"]

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_reward_df(base: str, symbol: str, frame: str, limit: Optional[int] = None) -> pd.DataFrame:
    path = os.path.join(base, symbol, frame, "train_log.csv")
    return read_csv_safely(path, limit=limit)


def load_trades_df(base: str, symbol: str, frame: str, limit: Optional[int] = None) -> pd.DataFrame:
    path = os.path.join(base, symbol, frame, "deep_rl_trades.csv")
    return read_csv_safely(path, limit=limit)


def load_steps_df(base: str, symbol: str, frame: str, limit: Optional[int] = None) -> pd.DataFrame:
    path = os.path.join(base, symbol, frame, "step_log.csv")
    return read_csv_safely(path, limit=limit)


def load_decisions_df(base: str, symbol: str, frame: str, limit: Optional[int] = None) -> pd.DataFrame:
    path = os.path.join(base, symbol, frame, "logs", "entry_decisions.jsonl")
    return read_jsonl_safely(path, limit=limit)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_equity(df: pd.DataFrame, equity_aliases: Sequence[str] = EQUITY_ALIASES) -> pd.Series:
    col = find_column(df, equity_aliases)
    if col is None:
        if "pnl" in df.columns:
            series = df["pnl"].astype(float).cumsum()
        else:
            return pd.Series(dtype=float)
    else:
        series = df[col].astype(float)
    series.name = "equity"
    return series


def compute_drawdown(eq: pd.Series) -> pd.Series:
    if eq.empty:
        return pd.Series(dtype=float)
    roll_max = eq.cummax()
    dd = (roll_max - eq) / roll_max.replace(0, np.nan)
    dd.name = "drawdown"
    return dd


def compute_sharpe(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    ret = eq.pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return float((ret.mean() / ret.std()) * np.sqrt(252))


def penalties_breakdown(step_df: pd.DataFrame, penalty_aliases: Sequence[str] = PENALTY_ALIASES) -> pd.DataFrame:
    cols = [c for c in penalty_aliases if c in step_df.columns]
    if not cols:
        return pd.DataFrame()
    return step_df[cols]


def signals_agg(decisions_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if decisions_df.empty or "signal" not in decisions_df.columns:
        return pd.DataFrame()
    agg = decisions_df.groupby("signal").size().sort_values(ascending=False).head(top_n)
    return agg.reset_index().rename(columns={0: "count"})

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def atomic_save(fig: plt.Figure, out_path: str, fmt: str = "png") -> None:
    _ensure_dir(out_path)
    tmp = out_path + ".tmp"
    fig.savefig(tmp, format=fmt, dpi=120)
    plt.close(fig)
    os.replace(tmp, out_path)


def plot_line(series: pd.Series, out_path: str, title: str = "") -> None:
    if series.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values)
    ax.set_title(title)
    fig.tight_layout()
    atomic_save(fig, out_path)


def plot_area(df: pd.DataFrame, out_path: str, title: str = "") -> None:
    if df.empty:
        return
    fig, ax = plt.subplots()
    df.plot.area(ax=ax, stacked=True)
    ax.set_title(title)
    fig.tight_layout()
    atomic_save(fig, out_path)


def plot_bar(df: pd.DataFrame, out_path: str, title: str = "") -> None:
    if df.empty:
        return
    fig, ax = plt.subplots()
    df.plot.bar(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    atomic_save(fig, out_path)


def table_to_image(df: pd.DataFrame, out_path: str, title: str = "") -> None:
    if df.empty:
        return
    fig, ax = plt.subplots()
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.scale(1, 1.5)
    ax.set_title(title)
    fig.tight_layout()
    atomic_save(fig, out_path)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def git_short_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"
