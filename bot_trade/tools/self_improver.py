import os
import json
import pandas as pd
from collections import defaultdict

KNOWLEDGE_FILE = "memory/knowledge_base.json"

def load_performance_data(step_log_path="results/step_log.csv"):
    if not os.path.exists(step_log_path):
        print("[STRATEGY] Step log not found.")
        return pd.DataFrame()
    df = pd.read_csv(step_log_path)
    return df[df["done"] == True]  # Only finished episodes

def extract_signal_stats(df):
    wins = defaultdict(int)
    losses = defaultdict(int)

    for _, row in df.iterrows():
        if not isinstance(row.get("signals_used"), str):
            continue
        signals = [sig.strip() for sig in row["signals_used"].split(";") if sig.strip()]
        if row.get("pnl", 0) > 0:
            for sig in signals:
                wins[sig] += 1
        elif row.get("pnl", 0) < 0:
            for sig in signals:
                losses[sig] += 1

    signal_stats = {}
    all_signals = set(wins.keys()) | set(losses.keys())
    for sig in all_signals:
        w = wins[sig]
        l = losses[sig]
        total = w + l
        win_rate = round(w / total, 3) if total > 0 else 0.0
        signal_stats[sig] = {
            "wins": w,
            "losses": l,
            "win_rate": win_rate
        }

    return wins, losses, signal_stats

def update_knowledge_base(frame, symbol, win_signals, loss_signals, stats):
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {}

    key = f"{symbol}_{frame}"
    kb.setdefault(key, {})

    for field, data in zip(["successful_signals", "failing_signals"], [win_signals, loss_signals]):
        kb[key].setdefault(field, {})
        for sig, count in data.items():
            prev = kb[key][field].get(sig, 0)
            kb[key][field][sig] = prev + count

    kb[key]["signal_stats"] = stats

    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"[KNOWLEDGE ✅] Updated knowledge base for {key} with {len(stats)} signals.")

def main():
    frame = os.getenv("CURRENT_FRAME", "1m")
    symbol = os.getenv("CURRENT_SYMBOL", "BTC")

    df = load_performance_data()
    if df.empty:
        print("[STRATEGY] No completed sessions to analyze.")
        return

    win_signals, loss_signals, signal_stats = extract_signal_stats(df)
    if signal_stats:
        update_knowledge_base(frame, symbol, win_signals, loss_signals, signal_stats)
    else:
        print("[STRATEGY] No signal statistics to update.")

if __name__ == "__main__":
    main()
