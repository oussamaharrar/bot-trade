# ai_core/strategy_learner.py
# Learns simple per-signal and per-frame stats and updates the unified KB
from __future__ import annotations
import os, json
from collections import defaultdict
from typing import Dict, Any
import pandas as pd

KB_FILE = "memory/knowledge_base_full.json"
STEP_LOG = os.getenv("STEP_LOG", os.path.join("results", "step_log.csv"))


def _load_json(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_performance_data(step_log_path: str = STEP_LOG) -> pd.DataFrame:
    if not os.path.exists(step_log_path):
        return pd.DataFrame()
    df = pd.read_csv(step_log_path)
    # Expect columns: episode_done, reward, frame, symbol, entry_signals (optional), ...
    done_col = "done" if "done" in df.columns else ("episode_done" if "episode_done" in df.columns else None)
    if done_col:
        df = df[df[done_col] == True]
    return df


def extract_signal_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df is None or df.empty:
        return {}
    wins = defaultdict(int)
    losses = defaultdict(int)
    counts = defaultdict(int)

    sig_col = "entry_signals"
    rew_col = "reward"

    for _, row in df.iterrows():
        reward = float(row.get(rew_col, 0.0) or 0.0)
        # entry_signals may be semi-colon concatenated string like "rsi_recovery;breakout_buy;"
        raw = str(row.get(sig_col, "")).strip()
        names = [n for n in raw.split(";") if n]
        if not names:
            continue
        for n in names:
            counts[n] += 1
            if reward > 0:
                wins[n] += 1
            elif reward < 0:
                losses[n] += 1

    out: Dict[str, Dict[str, float]] = {}
    for k in counts:
        c = max(1, counts[k])
        out[k] = {
            "count": float(c),
            "win_rate": float(wins[k]) / float(c),
            "loss_rate": float(losses[k]) / float(c),
        }
    return out


def extract_frame_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df is None or df.empty:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    if "frame" not in df.columns:
        return out
    if "reward" not in df.columns:
        return out
    g = df.groupby("frame")["reward"].agg(["count", "mean"]).reset_index()
    for _, r in g.iterrows():
        out[str(r["frame"])]= {"count": float(r["count"]), "reward_mean": float(r["mean"]) }
    return out


def update_kb_from_logs(step_log_path: str = STEP_LOG) -> Dict[str, Any]:
    kb = _load_json(KB_FILE)
    df = load_performance_data(step_log_path)
    sig_stats = extract_signal_stats(df)
    frame_stats = extract_frame_stats(df)

    kb.setdefault("strategy_memory", {})
    kb.setdefault("signals_memory", {})

    # Merge frame stats under strategy_memory
    for frame, st in frame_stats.items():
        kb["strategy_memory"].setdefault(frame, {})
        kb["strategy_memory"][frame].update(st)

    # Merge signal stats
    for sig, st in sig_stats.items():
        kb["signals_memory"].setdefault(sig, {})
        kb["signals_memory"][sig].update(st)

    _save_json(KB_FILE, kb)
    return kb


if __name__ == "__main__":
    kb = update_kb_from_logs(STEP_LOG)
    print("KB updated. sections:", list(kb.keys()))
