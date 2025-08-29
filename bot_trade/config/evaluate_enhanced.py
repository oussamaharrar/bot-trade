import os
import numpy as np
import pandas as pd
import strategy_features  # استيراد جميع دوال المؤشرات
from datetime import datetime

# ==== مؤشرات الأداء المالي ====
def sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=252):
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    std = returns.std()
    return excess / std if std else np.nan

def sortino_ratio(returns, risk_free_rate=0.04, periods_per_year=252):
    neg = returns[returns < 0]
    downside = neg.std()
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    return excess / downside if downside else np.nan

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    return drawdown.min()

def calmar_ratio(returns):
    md = max_drawdown(returns.cumsum())
    return returns.mean() / abs(md) if md else np.nan

def kelly_criterion(win_rate, avg_win, avg_loss):
    return win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss else np.nan

# ==== دمج المؤشرات المخصصة مع نتائج الصفقات ====
def annotate_with_indicators(trades_df):
    # توقع أن strategy_features يوفر دالة add_indicators(df)
    df = trades_df.copy()
    df = strategy_features.add_indicators(df)
    return df

# ==== التقييم الرئيسي ====
def evaluate_trades(trades_df, save_path="results/evaluation.csv", model_name=None, strategy=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if trades_df.empty or 'pnl' not in trades_df.columns:
        print("❌ بيانات غير صالحة للتقييم.")
        return

    trades = trades_df.dropna(subset=['pnl'])
    returns = trades['pnl']
    wins = returns[returns > 0]
    losses = -returns[returns < 0]

    # الحساب الأساسي
    metrics = {
        'timestamp': datetime.now(),
        'model': model_name or 'unknown',
        'strategy': strategy or 'N/A',
        'total_profit_%': returns.sum() * 100,
        'win_rate_%': len(wins) / len(returns) * 100,
        'max_drawdown_%': max_drawdown(returns.cumsum()) * 100,
        'sharpe': sharpe_ratio(returns),
        'sortino': sortino_ratio(returns),
        'calmar': calmar_ratio(returns),
        'kelly_%': kelly_criterion(len(wins)/len(returns), wins.mean(), losses.mean()) * 100
    }

    # إضافة تحليل مؤشرات مخصصة
    annotated = annotate_with_indicators(trades)
    # مثال: حساب متوسط الربح/التراجع لكل مؤشر
    indicator_cols = [col for col in annotated.columns if col not in ['pnl']]
    for col in indicator_cols:
        metrics[f'{col}_corr'] = np.corrcoef(annotated['pnl'], annotated[col])[0,1]

    dfm = pd.DataFrame([metrics])
    write_mode = 'a' if os.path.exists(save_path) else 'w'
    dfm.to_csv(save_path, mode=write_mode, header=(write_mode=='w'), index=False)

    print("\n✅ نتائج التقييم:")
    print(dfm.T.round(4))

if __name__ == '__main__':
    print("⚠️ لا تنفيذ مباشر. استدعِ evaluate_trades(df, path, model, strategy) من سكربت آخر.")
