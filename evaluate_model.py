import os
from datetime import datetime
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import json

REPORTS_DIR = 'reports'
MODEL_PATH = 'models/trained_model.pkl'
EVAL_LOG = 'model_evaluation_log.csv'

for d in ['models', 'results', REPORTS_DIR, 'logs']:
    os.makedirs(d, exist_ok=True)

def evaluate():
    df = pd.read_csv('training_dataset.csv')
    df = df.dropna(subset=['pnl_class'])
    X = df.drop(columns=['pnl_class']).select_dtypes(include=['number'])
    y = df['pnl_class']

    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, expected_cols = joblib.load(MODEL_PATH)
    X_test = X_test[expected_cols]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(classification_report(y_test, y_pred))

    test_df = df.iloc[y_test.index].copy()
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    if 'total_value' in test_df.columns:
        test_df['returns'] = test_df['total_value'].pct_change().fillna(0)
    elif 'pnl' in test_df.columns:
        test_df['returns'] = test_df['pnl']
    else:
        test_df['returns'] = 0

    def compute_sharpe(returns):
        return (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(len(returns))

    sharpe = compute_sharpe(test_df['returns'])

    # Daily/Weekly/Monthly Sharpe Ratios
    test_df = test_df.set_index('timestamp')
    daily_sharpe = compute_sharpe(test_df['returns'].resample('1D').sum())
    weekly_sharpe = compute_sharpe(test_df['returns'].resample('1W').sum())
    monthly_sharpe = compute_sharpe(test_df['returns'].resample('1M').sum())

    cumulative = (1 + test_df['returns']).cumprod()
    roll_max = cumulative.cummax()
    drawdown = (cumulative - roll_max) / roll_max
    cumulative_drawdown = drawdown.cumsum()
    max_drawdown = drawdown.min()
    win_rate = (test_df['returns'] > 0).mean()

    print(f'Sharpe Ratio: {sharpe:.4f}')
    print(f'Daily Sharpe: {daily_sharpe:.4f}')
    print(f'Weekly Sharpe: {weekly_sharpe:.4f}')
    print(f'Monthly Sharpe: {monthly_sharpe:.4f}')
    print(f'Max Drawdown: {max_drawdown:.4f}')
    print(f'Win Rate: {win_rate:.2%}')

    log_exists = os.path.exists(EVAL_LOG)
    with open(EVAL_LOG, 'a') as f:
        if not log_exists:
            f.write('timestamp,accuracy,f1_score,sharpe,sharpe_daily,sharpe_weekly,sharpe_monthly,max_drawdown,win_rate\n')
        f.write(
            f"{datetime.now()},{accuracy:.4f},{f1:.4f},{sharpe:.4f},{daily_sharpe:.4f},{weekly_sharpe:.4f},{monthly_sharpe:.4f},{max_drawdown:.4f},{win_rate:.4f}\n"
        )

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M')
    cm_path = os.path.join(REPORTS_DIR, f'conf_matrix_{ts}.png')
    trend_path = os.path.join(REPORTS_DIR, f'trend_{ts}.png')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Permutation importance for feature sensitivity
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='f1_weighted')
    perm_df = pd.DataFrame({'Feature': expected_cols, 'Importance': perm.importances_mean})
    perm_df = perm_df.sort_values('Importance', ascending=False)
    perm_path = os.path.join(REPORTS_DIR, f'feature_sensitivity_{ts}.png')
    plt.figure(figsize=(8,5))
    sns.barplot(data=perm_df, x='Importance', y='Feature', color='blue')
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig(perm_path)
    plt.close()

    pnl_path = None
    if 'total_value' in test_df.columns:
        pnl_path = os.path.join(REPORTS_DIR, f'pnl_{ts}.png')
        plt.figure(figsize=(8,4))
        plt.plot(test_df.index, cumulative)
        plt.title('Cumulative PnL')
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative Return')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(pnl_path)
        plt.close()

    # Drawdown curves
    drawdown_path = os.path.join(REPORTS_DIR, f'drawdown_{ts}.png')
    plt.figure(figsize=(8,4))
    plt.plot(test_df.index, drawdown, label='Drawdown')
    plt.plot(test_df.index, cumulative_drawdown, label='Cumulative Drawdown')
    plt.title('Drawdown')
    plt.xlabel('Timestamp')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(drawdown_path)
    plt.close()

    if os.path.exists(EVAL_LOG):
        try:
            log_df = pd.read_csv(EVAL_LOG)
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            log_df = log_df.sort_values('timestamp')
            plt.figure(figsize=(8,4))
            plt.plot(log_df['timestamp'], log_df['accuracy'], label='Accuracy', marker='s')
            plt.plot(log_df['timestamp'], log_df['f1_score'], label='F1 Score', marker='o')
            plt.xticks(rotation=45)
            plt.title('Model Performance Trend')
            plt.legend()
            plt.tight_layout()
            plt.savefig(trend_path)
            plt.close()
        except Exception as e:
            print(f'Could not generate trend plot: {e}')
            trend_path = None
    else:
        trend_path = None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Model Evaluation Report', ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generated: {datetime.now()}', ln=1)
    pdf.cell(0, 10, f'Accuracy: {accuracy:.4f}', ln=1)
    pdf.cell(0, 10, f'F1 Score: {f1:.4f}', ln=1)
    pdf.cell(0, 10, f'Sharpe Ratio: {sharpe:.4f}', ln=1)
    pdf.cell(0, 10, f'Sharpe (Daily): {daily_sharpe:.4f}', ln=1)
    pdf.cell(0, 10, f'Sharpe (Weekly): {weekly_sharpe:.4f}', ln=1)
    pdf.cell(0, 10, f'Sharpe (Monthly): {monthly_sharpe:.4f}', ln=1)
    pdf.cell(0, 10, f'Max Drawdown: {max_drawdown:.4f}', ln=1)
    pdf.cell(0, 10, f'Win Rate: {win_rate:.2%}', ln=1)
    pdf.ln(5)
    pdf.image(cm_path, w=170)
    if trend_path:
        pdf.ln(5)
        pdf.image(trend_path, w=170)
    if pnl_path:
        pdf.ln(5)
        pdf.image(pnl_path, w=170)
    pdf.ln(5)
    pdf.image(drawdown_path, w=170)
    pdf.ln(5)
    pdf.image(perm_path, w=170)
    report_path = os.path.join(REPORTS_DIR, f'report_{ts}.pdf')
    pdf.output(report_path)
    print(f'Report saved to {report_path}')

    summary = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'f1_score': f1,
        'sharpe': sharpe,
        'sharpe_daily': daily_sharpe,
        'sharpe_weekly': weekly_sharpe,
        'sharpe_monthly': monthly_sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'confusion_matrix_path': cm_path,
        'trend_path': trend_path,
        'pnl_path': pnl_path,
        'drawdown_path': drawdown_path,
        'feature_sensitivity_path': perm_path,
        'report_path': report_path,
    }
    summary_path = os.path.join(REPORTS_DIR, f'summary_{ts}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to {summary_path}')

if __name__ == '__main__':
    evaluate()
