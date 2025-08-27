"""Simple Streamlit dashboard for monitoring training outputs.

The dashboard reads CSV/JSONL logs from ``results/<SYMBOL>/<FRAME>/logs``
 and visualises basic curves such as reward and trades.  The goal is to
 provide a light‑weight real‑time view without relying on external
 services.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return items


def main() -> None:
    st.title("Bot‑Trade Dashboard")

    base = Path(st.sidebar.text_input("Results root", "results"))
    symbol = st.sidebar.text_input("Symbol", "BTCUSDT").upper()
    frame = st.sidebar.text_input("Frame", "1m")
    logs_dir = base / symbol / frame / "logs"
    st.sidebar.write(f"Logs dir: {logs_dir}")

    reward_df = _load_csv(logs_dir / "reward.csv")
    trades_df = _load_csv(logs_dir / "deep_rl_trades.csv")
    bench_df = _load_csv(logs_dir / "benchmark.log")
    decisions = _load_jsonl(logs_dir / "entry_decisions.jsonl")

    if not reward_df.empty:
        st.subheader("Reward curve")
        try:
            reward_df = reward_df.set_index("step")
        except Exception:
            pass
        st.line_chart(reward_df["avg_reward"])

    if not trades_df.empty:
        st.subheader("Trade log (last 50)")
        st.dataframe(trades_df.tail(50))

    if decisions:
        st.subheader("Recent entry decisions")
        st.json(decisions[-5:])

    if not bench_df.empty:
        st.subheader("Benchmark stats")
        st.dataframe(bench_df.tail(20))


if __name__ == "__main__":  # pragma: no cover - manual run
    main()
