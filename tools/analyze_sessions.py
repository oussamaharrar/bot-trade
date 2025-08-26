import argparse, os, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def load_trades_or_steps(results_dir="results"):
    cand = []
    for f in os.listdir(results_dir):
        if f.endswith(".csv") and ("step_log" in f or "deep_rl" in f or "trades" in f):
            cand.append(os.path.join(results_dir, f))
    dfs = [pd.read_csv(p) for p in cand] if cand else []
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def make_equity_curve(df):
    if "total_value" in df.columns:
        eq = df["total_value"].astype(float)
    else:
        pnl = df.get("pnl", pd.Series(np.zeros(len(df))))
        eq = pnl.cumsum()
    return pd.Series(eq.values, name="equity")

def save_curve(eq, frame, model):
    os.makedirs("reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    eq.to_csv(f"results/equity_curve_{frame}_{model}.csv", index=False)
    plt.figure()
    plt.plot(eq.values)
    plt.title(f"Equity Curve — {frame} — {model}")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(f"reports/equity_curve_{frame}_{model}.png", dpi=120)

def main():
    frame = os.getenv("CURRENT_FRAME", "unknown")
    model = os.getenv("CURRENT_MODEL", "deep_rl")
    results_dir = "results"

    df = load_trades_or_steps(results_dir)
    if df.empty:
        print("No results to analyze."); return

    eq = make_equity_curve(df)
    save_curve(eq, frame, model)
    ret = pd.Series(eq).pct_change().dropna()
    sharpe = (ret.mean()/ret.std()) * np.sqrt(252) if ret.std() > 0 else 0.0
    roll_max = eq.cummax()
    dd = (roll_max - eq) / roll_max.replace(0, np.nan)
    maxdd = float(dd.max())
    wins = (df.get("pnl", pd.Series(0)).astype(float) > 0).sum()
    total = len(df)
    winrate = wins / total if total else 0.0

    with open("reports/leaderboard.md", "a", encoding="utf-8") as f:
        f.write(f"| {frame} | {model} | Sharpe {sharpe:.3f} | MaxDD {maxdd:.2%} | WinRate {winrate:.2%} |\n")

    print(f"✅ جلسة التحليل مكتملة للفريم: {frame} — النموذج: {model}")

if __name__ == "__main__":
    main()
