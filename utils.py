import os
import pandas as pd

MODEL_DIR = 'models'
AGENT_DIR = 'agents'
RESULTS_DIR = 'results'
DATA_DIR = '.'
REPORTS_DIR = 'reports'


def list_models() -> list:
    models = []
    if os.path.isdir(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith('.pkl'):
                models.append(f)
    if os.path.isdir(AGENT_DIR):
        for f in os.listdir(AGENT_DIR):
            if f.endswith('.zip'):
                models.append(f)
    return models


def last_trades(n: int = 20) -> pd.DataFrame:
    if not os.path.isdir(RESULTS_DIR):
        return pd.DataFrame()
    files = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
    if not files:
        return pd.DataFrame()
    latest = max(files, key=os.path.getmtime)
    try:
        return pd.read_csv(latest).tail(n)
    except Exception:
        return pd.DataFrame()


def list_reports() -> list:
    if not os.path.isdir(REPORTS_DIR):
        return []
    return [os.path.join(REPORTS_DIR, f) for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf') or f.endswith('.csv')]

