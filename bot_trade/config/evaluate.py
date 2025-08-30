import os
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime


def sharpe_ratio(returns, rf=0.04, periods=252):
    excess = returns.mean() - (rf / periods)
    std = returns.std()
    return excess / std if std else 0.0

def sortino_ratio(returns, rf=0.04, periods=252):
    downside = returns[returns < 0].std()
    excess = returns.mean() - (rf / periods)
    return excess / downside if downside else 0.0

def max_drawdown(series):
    peak = series.cummax()
    drawdown = (series - peak)
    return drawdown.min()

def calmar_ratio(returns):
    md = max_drawdown(returns.cumsum())
    return returns.mean() / abs(md) if md else 0.0

def evaluate_trades(trades_df: pd.DataFrame, model_name: str, save_path: str = "results/evaluation.csv", additional_info: Optional[dict] = None):
    if trades_df.empty or 'pnl' not in trades_df:
        print("❌ لا يوجد بيانات pnl لتقييم النموذج.")
        return

    r = trades_df['pnl'].dropna()
    wins = r[r > 0]
    losses = -r[r < 0]

    stats = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "model": model_name,
        "total_profit_%": r.sum() * 100,
        "mean_reward": r.mean(),
        "win_rate_%": len(wins) / len(r) * 100,
        "max_drawdown_%": max_drawdown(r.cumsum()) * 100,
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "calmar": calmar_ratio(r),
        "kelly_%": ((len(wins)/len(r)) - (1 - len(wins)/len(r)) / (wins.mean() / losses.mean()) if losses.mean() else 0.0) * 100
    }

    if additional_info:
        stats.update(additional_info)

    df = pd.DataFrame([stats])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, index=False)

    print("\n📊 تقييم النموذج:")
    print(df.T.round(4))

if __name__ == '__main__':
    print("⚠️ هذا الملف مخصص للاستدعاء من داخل سكربتات التدريب أو التشغيل.")
