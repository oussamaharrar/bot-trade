import os
import json
from typing import Dict

import numpy as np
import pandas as pd

from .analytics_common import (
    load_steps_df,
    load_reward_df,
    compute_equity,
    compute_drawdown,
    compute_sharpe,
    penalties_breakdown,
)


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


def analyse_risk(base: str, symbol: str, frame: str) -> Dict[str, object]:
    risk_path = os.path.join(base, symbol, frame, "logs", "risk.log")
    risk_df = pd.read_csv(risk_path) if os.path.exists(risk_path) else pd.DataFrame()
    steps = load_steps_df(base, symbol, frame)
    rewards = load_reward_df(base, symbol, frame)
    equity = compute_equity(steps)
    drawdown = compute_drawdown(equity)
    sharpe = compute_sharpe(equity)
    penalties = penalties_breakdown(steps)
    reasons = risk_df['reason'].value_counts().to_dict() if not risk_df.empty else {}
    wins = (steps.get("pnl", pd.Series(0)).astype(float) > 0).sum()
    total = len(steps)
    winrate = wins / total if total else 0.0
    maxdd = float(drawdown.max()) if not drawdown.empty else 0.0
    return {
        "risk": risk_df,
        "steps": steps,
        "rewards": rewards,
        "equity": equity,
        "drawdown": drawdown,
        "penalties": penalties,
        "sharpe": sharpe,
        "winrate": winrate,
        "maxdd": maxdd,
        "reasons": reasons,
    }


def main():
    frame = os.getenv("CURRENT_FRAME", "unknown")
    symbol = os.getenv("CURRENT_SYMBOL", "BTCUSDT")
    base = os.getenv("RESULTS_BASE", "results")

    out = analyse_risk(base, symbol, frame)
    risk_df = out.get("risk", pd.DataFrame())
    sharpe = out.get("sharpe", 0.0)
    maxdd = out.get("maxdd", 0.0)
    winrate = out.get("winrate", 0.0)
    reasons = out.get("reasons", {})

    # simple tuning hints
    hints = {
        "increase_vol_penalty": reasons.get("High market volatility", 0) > 10,
        "freeze_on_drawdown": reasons.get("Persistent drawdown", 0) > 5 or maxdd > 0.3,
        "boost_trend_bonus": reasons.get("Trend-follow signal with strong ADX", 0) > 3,
        "penalize_mean_reversion": reasons.get("Mean-reversion signal penalized (high ADX)", 0) > 3,
        "slow_down_training": sharpe < 0.3 and winrate < 0.4,
    }

    os.makedirs("reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    with open(f"reports/analysis_summary_{frame}.md", "w", encoding="utf-8") as f:
        f.write(render_markdown_summary(frame, sharpe, maxdd, winrate, reasons))
    risk_df.to_csv(f"results/enriched_risk_log_{frame}.csv", index=False)
    with open(f"results/tuning_hints_{frame}.json", "w", encoding="utf-8") as f:
        json.dump(hints, f, indent=2, ensure_ascii=False)
    print(f"✅ analysis frame done !: {frame}")


if __name__ == "__main__":
    main()
