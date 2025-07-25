import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    loaded = joblib.load(MODEL_PATH)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(classification_report(y_test, y_pred))

    test_df = df.loc[X_test.index]
    returns = pd.to_numeric(test_df.get('pnl', pd.Series(index=test_df.index, data=0)), errors='coerce').fillna(0)
    strategy_returns = returns.where(y_pred == 1, 0)

    win_rate = float((strategy_returns > 0).mean())
    if strategy_returns.std() != 0:
        sharpe_ratio = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(len(strategy_returns)))
    else:
        sharpe_ratio = 0.0
    cum_returns = strategy_returns.cumsum()
    max_drawdown = float((cum_returns - cum_returns.cummax()).min())

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M')
    cm_path = os.path.join(REPORTS_DIR, f'conf_matrix_{ts}.png')
    trend_path = os.path.join(REPORTS_DIR, f'trend_{ts}.png')
    pnl_path = os.path.join(REPORTS_DIR, f'cumulative_pnl_{ts}.png')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    if os.path.exists(EVAL_LOG):
        log_df = pd.read_csv(EVAL_LOG, on_bad_lines='skip')
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
    else:
        trend_path = None

    if 'total_value' in df.columns:
        df_sorted = df.sort_values('timestamp')
        pnl_series = df_sorted['total_value'] - df_sorted['total_value'].iloc[0]
        plt.figure(figsize=(8,4))
        plt.plot(pd.to_datetime(df_sorted['timestamp']), pnl_series)
        plt.title('Cumulative PnL')
        plt.xlabel('Timestamp')
        plt.ylabel('PnL')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(pnl_path)
        plt.close()
    else:
        pnl_path = None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Model Evaluation Report', ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generated: {datetime.now()}', ln=1)
    pdf.cell(0, 10, f'Accuracy: {accuracy:.4f}', ln=1)
    pdf.cell(0, 10, f'F1 Score: {f1:.4f}', ln=1)
    pdf.cell(0, 10, f'Sharpe Ratio: {sharpe_ratio:.4f}', ln=1)
    pdf.cell(0, 10, f'Max Drawdown: {max_drawdown:.4f}', ln=1)
    pdf.cell(0, 10, f'Win Rate: {win_rate:.4f}', ln=1)
    pdf.ln(5)
    pdf.image(cm_path, w=170)
    if trend_path:
        pdf.ln(5)
        pdf.image(trend_path, w=170)
    if pnl_path:
        pdf.ln(5)
        pdf.image(pnl_path, w=170)
    report_path = os.path.join(REPORTS_DIR, f'report_{ts}.pdf')
    pdf.output(report_path)
    print(f'Report saved to {report_path}')

    log_exists = os.path.exists(EVAL_LOG)
    with open(EVAL_LOG, 'a') as f:
        if not log_exists:
            f.write('timestamp,accuracy,f1_score,sharpe_ratio,max_drawdown,win_rate,model_path\n')
        f.write(f"{datetime.now()},{accuracy:.4f},{f1:.4f},{sharpe_ratio:.4f},{max_drawdown:.4f},{win_rate:.4f},{MODEL_PATH}\n")

if __name__ == '__main__':
    evaluate()
