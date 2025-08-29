"""Streamlit dashboard for monitoring training outputs (read-only).

Run with:
    streamlit run ai_core/ai_dashboard.py
or from project root:
    streamlit run dashboard.py
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from .dashboard_io import (
    load_decisions,
    load_training_metrics,
    load_trades,
    load_benchmark,
    load_knowledge,
)


def first_available_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    low = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in low:
            return df.columns[low.index(cand.lower())]
    return None


def ensure_avg_reward(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    aliases = ["avg_reward", "ep_rew_mean", "mean_reward", "episode_reward", "reward"]
    col = first_available_column(df, aliases)
    if col is None:
        return df
    if col != "avg_reward":
        df["avg_reward"] = pd.to_numeric(df[col], errors="coerce")
    if "avg_reward" in df.columns and "ep_rew_mean" not in df.columns:
        df["avg_reward"] = df["avg_reward"].rolling(window=100, min_periods=10).mean()
    return df


def _cumulative_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    try:
        pnl = trades.get("pnl")
        if pnl is None:
            return pd.Series(dtype=float)
        return pnl.cumsum()
    except Exception:
        return pd.Series(dtype=float)


def main() -> None:
    st.set_page_config(page_title="Bot-Trade Dashboard", layout="wide")
    st.title("Bot-Trade Dashboard")

    with st.sidebar:
        base = st.text_input("Results root", "results")
        symbol = st.text_input("Symbol", "BTCUSDT").upper()
        frame = st.text_input("Frame", "1m")
        refresh = st.number_input("Refresh (sec)", min_value=2, max_value=60, value=5)
        st.write("Reading from", Path(base) / symbol / frame)
        try:  # streamlit >=1.19
            st.autorefresh(interval=int(refresh * 1000), key="refresh")
        except Exception:
            pass

    metrics: Dict[str, pd.DataFrame] = load_training_metrics(symbol, frame)
    trades_df = load_trades(symbol, frame)
    benchmark_df = load_benchmark(symbol, frame)
    decisions_df = load_decisions(symbol, frame)
    knowledge = load_knowledge()

    if not metrics.get("train_log", pd.DataFrame()).empty:
        st.subheader("Training log (tail)")
        st.dataframe(metrics["train_log"].tail(20))

    reward_df = metrics.get("reward", pd.DataFrame())
    if reward_df.empty:
        reward_df = metrics.get("step_log", pd.DataFrame())
    if reward_df.empty:
        reward_df = metrics.get("train_log", pd.DataFrame())
    reward_df = ensure_avg_reward(reward_df)
    if not reward_df.empty and "avg_reward" in reward_df.columns:
        st.subheader("Reward curve")
        try:
            reward_df = reward_df.set_index("step")
        except Exception:
            pass
        st.line_chart(reward_df["avg_reward"])
    elif not reward_df.empty:
        st.subheader("Reward curve (fallback)")
        st.info("avg_reward missing; plotting numeric columns")
        num = reward_df.select_dtypes(include="number")
        if not num.empty:
            st.line_chart(num)

    if not trades_df.empty:
        st.subheader("Equity curve")
        pnl_series = _cumulative_pnl(trades_df)
        if not pnl_series.empty:
            st.line_chart(pnl_series)
        st.subheader("Recent trades")
        st.dataframe(trades_df.tail(50))

    if not decisions_df.empty:
        st.subheader("Recent entry decisions")
        st.dataframe(decisions_df.tail(20))

    if not benchmark_df.empty:
        st.subheader("System benchmark (tail)")
        st.dataframe(benchmark_df.tail(20))

    # Knowledge base quick view
    if knowledge:
        st.subheader("Top signals (win_rate >= 0.5, n>=50)")
        sigs = knowledge.get("signals_memory", {})
        rows = []
        for name, stats in sigs.items():
            count = float(stats.get("count", 0))
            win = float(stats.get("win_rate", 0))
            if count >= 50 and win >= 0.5:
                rows.append({"signal": name, "win_rate": win, "count": count})
        if rows:
            df = pd.DataFrame(rows).sort_values("win_rate", ascending=False)
            st.table(df)


if __name__ == "__main__":  # pragma: no cover
    main()
