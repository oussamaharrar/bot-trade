# 🔁 Unified Logic: Learn from Failures + Amplify Success

# 📁 File: tools/session_learning_engine.py
# 📌 Goal: Analyze failed and successful sessions to extract intelligent recommendations

import json
import os
from pathlib import Path

import pandas as pd

from bot_trade.config.rl_paths import ensure_utf8, memory_dir


def load_logs():
    train_log = pd.read_csv('results/train_log.csv')
    risk_log = pd.read_csv('logs/risk.log')
    return train_log, risk_log


def extract_failed_sessions(train_log, sharpe_thresh=0.2, winrate_thresh=0.4):
    return train_log[(train_log['sharpe'] < sharpe_thresh) & (train_log['win_rate'] < winrate_thresh)]


def extract_successful_sessions(train_log, sharpe_thresh=1.0, winrate_thresh=0.6):
    return train_log[(train_log['sharpe'] > sharpe_thresh) | (train_log['win_rate'] > winrate_thresh)]


def analyze_failures(failed_df, risk_log):
    # Analyze common failure reasons
    risk_counts = risk_log['reason'].value_counts().to_dict()

    suggestions = {
        "disable_signals": [],
        "adjust_risk": True if 'High market volatility' in risk_counts else False,
        "review_frames": failed_df['frame'].value_counts().head(3).index.tolist(),
        "common_fail_reasons": risk_counts
    }
    return suggestions


def analyze_successes(success_df):
    frames = success_df['frame'].value_counts().head(3).index.tolist()
    coins = success_df['symbol'].value_counts().head(3).index.tolist() if 'symbol' in success_df.columns else []

    patterns = {
        "preferred_frames": frames,
        "preferred_symbols": coins,
        "avg_sharpe": success_df['sharpe'].mean(),
        "avg_winrate": success_df['win_rate'].mean()
    }
    return patterns


def save_json(data, path: Path) -> None:
    with ensure_utf8(path, csv_newline=False) as f:
        json.dump(data, f, indent=2)


def save_markdown(failure_data, success_data, path: Path = Path('reports/session_learning_summary.md')):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ensure_utf8(path, csv_newline=False) as f:
        f.write("# 📊 Session Learning Report\n\n")
        f.write("## ❌ Failure Analysis\n")
        for key, value in failure_data.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n## ✅ Success Patterns\n")
        for key, value in success_data.items():
            f.write(f"- **{key}**: {value}\n")


def main():
    mem = memory_dir()
    train_log, risk_log = load_logs()

    failed = extract_failed_sessions(train_log)
    success = extract_successful_sessions(train_log)

    fail_analysis = analyze_failures(failed, risk_log)
    success_patterns = analyze_successes(success)

    save_json(fail_analysis, mem / 'failure_insights.json')
    save_json(success_patterns, mem / 'success_patterns.json')
    save_markdown(fail_analysis, success_patterns)

    print("✅ Session analysis complete. Recommendations saved to memory/ and report to reports/.")


if __name__ == '__main__':
    main()
