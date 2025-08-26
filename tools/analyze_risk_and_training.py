import pandas as pd
import numpy as np
import json
import os
import argparse

def render_markdown_summary(frame, sharpe, maxdd, winrate, reasons):
    summary = f"""
# تقرير تحليل أداء النموذج ({frame})

- Sharpe Ratio: **{sharpe:.3f}**
- Max Drawdown: **{maxdd:.2%}**
- Win Rate: **{winrate:.2%}**

## أكثر الأسباب فعالية في ضبط المخاطر:
"""
    for r, c in reasons.items():
        summary += f"- {r}: {c} مرة\n"
    return summary

def main():
    frame = os.getenv("CURRENT_FRAME", "unknown")

    logs_path = "logs/risk.log"
    steps_path = "results/step_log.csv"
    train_path = "results/train_log.csv"

    risk = pd.read_csv(logs_path) if os.path.exists(logs_path) else pd.DataFrame()
    steps = pd.read_csv(steps_path) if os.path.exists(steps_path) else pd.DataFrame()
    sessions = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame()

    # --- الأسباب الأكثر تكرارًا ---
    reasons = risk['reason'].value_counts().to_dict() if not risk.empty else {}
    freezes = sum("freeze" in r.lower() for r in reasons.keys())

    # --- الحسابات ---
    eq = steps.get("total_value", pd.Series(np.zeros(len(steps))))
    ret = eq.pct_change().dropna()
    sharpe = (ret.mean()/ret.std()) * np.sqrt(252) if ret.std() > 0 else 0.0
    roll_max = eq.cummax()
    dd = (roll_max - eq) / roll_max.replace(0, np.nan)
    maxdd = float(dd.max())
    wins = (steps.get("pnl", pd.Series(0)).astype(float) > 0).sum()
    total = len(steps)
    winrate = wins / total if total else 0.0

    # --- توصيات تعليم ---
    hints = {
        "increase_vol_penalty": reasons.get("High market volatility", 0) > 10,
        "freeze_on_drawdown": reasons.get("Persistent drawdown", 0) > 5 or maxdd > 0.3,
        "boost_trend_bonus": reasons.get("Trend-follow signal with strong ADX", 0) > 3,
        "penalize_mean_reversion": reasons.get("Mean-reversion signal penalized (high ADX)", 0) > 3,
        "slow_down_training": sharpe < 0.3 and winrate < 0.4
    }

    os.makedirs("reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Save markdown summary
    with open(f"reports/analysis_summary_{frame}.md", "w", encoding="utf-8") as f:
        f.write(render_markdown_summary(frame, sharpe, maxdd, winrate, reasons))

    # Save enriched CSV
    risk.to_csv(f"results/enriched_risk_log_{frame}.csv", index=False)

    # Save hints
    with open(f"results/tuning_hints_{frame}.json", "w", encoding="utf-8") as f:
        json.dump(hints, f, indent=2, ensure_ascii=False)

    print(f"✅ analysis frame done !: {frame}")

if __name__ == "__main__":
    main()
