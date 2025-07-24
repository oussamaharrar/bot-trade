import os
from datetime import datetime
import pandas as pd
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
    X = df.drop(columns=['pnl_class']).select_dtypes(include=['number'])
    y = df['pnl_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(classification_report(y_test, y_pred))

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

    if os.path.exists(EVAL_LOG):
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
    pdf.ln(5)
    pdf.image(cm_path, w=170)
    if trend_path:
        pdf.ln(5)
        pdf.image(trend_path, w=170)
    report_path = os.path.join(REPORTS_DIR, f'report_{ts}.pdf')
    pdf.output(report_path)
    print(f'Report saved to {report_path}')

if __name__ == '__main__':
    evaluate()
