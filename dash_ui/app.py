import os
import json
import subprocess
import tempfile
import zipfile

import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(ROOT, 'results')
MEMORY_FILE = os.path.join(ROOT, 'memory', 'memory.json')
EVAL_LOG = os.path.join(ROOT, 'model_evaluation_log.csv')
REPORTS_DIR = os.path.join(ROOT, 'reports')
MODELS_DIR = os.path.join(ROOT, 'models')

st.set_page_config(page_title='Trading Bot Dashboard', layout='wide')

st.sidebar.header('Settings')
risk_threshold = st.sidebar.slider('Risk Threshold', 0.0, 1.0, 0.5, 0.01)
agent_type = st.sidebar.selectbox('Agent Type', ['rule', 'ml', 'rl'])
policy = st.sidebar.text_input('Policy', 'PPO')
timesteps = st.sidebar.number_input('Timesteps', min_value=1000, value=10000, step=1000)
data_source = st.sidebar.text_input('Data Source', 'training_dataset.csv')
model_path = st.sidebar.text_input('Model Path', os.path.join('models', 'trained_model.pkl'))

st.sidebar.markdown('---')
if st.sidebar.button('Train ML Model'):
    subprocess.run(['python', os.path.join(ROOT, 'autolearn.py')])
if st.sidebar.button('Evaluate Model'):
    subprocess.run(['python', os.path.join(ROOT, 'evaluate_model.py')])
if st.sidebar.button('Train RL Agent'):
    subprocess.run(['python', os.path.join(ROOT, 'train_rl.py')])
if st.sidebar.button('Run RL Agent'):
    subprocess.run(['python', os.path.join(ROOT, 'run_rl_agent.py')])

st.title('Trading Bot Dashboard')

# Live logs
st.header('Live Logs')

def latest_result_file():
    if not os.path.isdir(RESULTS_DIR):
        return None
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('run_') and f.endswith('.csv')]
    if not files:
        return None
    return os.path.join(RESULTS_DIR, sorted(files)[-1])

res_file = latest_result_file()
if res_file and os.path.exists(res_file):
    st.subheader('Latest Trade Log')
    df = pd.read_csv(res_file).tail(20)
    st.dataframe(df)
else:
    st.info('No trade logs found.')

if os.path.exists(MEMORY_FILE):
    st.subheader('Memory')
    with open(MEMORY_FILE) as f:
        mem = json.load(f)
    st.json(mem)

# Leaderboard
st.header('Model Leaderboard')
if os.path.exists(EVAL_LOG):
    eval_df = pd.read_csv(EVAL_LOG)
    cols = [c for c in ['timestamp', 'f1_score', 'sharpe', 'accuracy', 'model_path'] if c in eval_df.columns]
    eval_df = eval_df.sort_values('f1_score', ascending=False)
    st.dataframe(eval_df[cols])
else:
    st.info('No evaluation log found.')

# Reports
st.header('Reports')
if os.path.isdir(REPORTS_DIR):
    reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')]
    for r in sorted(reports, reverse=True):
        path = os.path.join(REPORTS_DIR, r)
        with open(path, 'rb') as f:
            data = f.read()
        st.download_button(label=f'Download {r}', data=data, file_name=r)
else:
    st.info('No reports found.')

# File upload
st.header('Add Training Data')
uploaded = st.file_uploader('Upload CSV', type='csv')
if uploaded is not None:
    df_new = pd.read_csv(uploaded)
    df_new.to_csv(os.path.join(ROOT, 'training_dataset.csv'), mode='a', index=False, header=False)
    st.success('Training data appended.')

# Export models and reports
st.header('Export Models & Reports')
if st.button('Create Export Archive'):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        with zipfile.ZipFile(tmp.name, 'w') as z:
            for folder in [MODELS_DIR, REPORTS_DIR]:
                if os.path.isdir(folder):
                    for file in os.listdir(folder):
                        z.write(os.path.join(folder, file), arcname=os.path.join(os.path.basename(folder), file))
        with open(tmp.name, 'rb') as f:
            bytes_data = f.read()
    st.download_button('Download Export', bytes_data, file_name='export.zip')
