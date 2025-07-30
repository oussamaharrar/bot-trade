import os
import pandas as pd
import streamlit as st
from utils import last_trades


def render():
    st.header("Live Analytics")
    st.caption("Monitor model performance in real time")

    log_file = 'model_evaluation_log.csv'
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        st.line_chart(df[['f1_score', 'accuracy']])
    else:
        st.info('No evaluation log found.')

    trades = last_trades(10)
    if not trades.empty:
        st.subheader('Last Trades')
        st.dataframe(trades)
    else:
        st.info('No trades logged yet.')

    st.download_button('Download CSV', log_file, disabled=not os.path.exists(log_file))
    st.markdown("---")
    st.subheader("Market Activity Heatmap")
    st.info("Placeholder for market heatmap visualization")
    st.subheader("Model vs Market")
    st.info("Placeholder for benchmark comparison")


import streamlit as st
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path('results')


def load_runs():
    runs = []
    for f in sorted(RESULTS_DIR.glob('run_*.csv')):
        try:
            df = pd.read_csv(f)
            df['run'] = f.name
            runs.append(df)
        except Exception:
            continue
    if runs:
        return pd.concat(runs, ignore_index=True)
    return pd.DataFrame()


def render():
    st.header('Live Analytics')
    st.caption('View performance metrics of your agents.')

    df = load_runs()
    if df.empty:
        st.info('No run data available.')
        return

    agent = st.selectbox('Select run', sorted(df['run'].unique()))
    data = df[df['run'] == agent]
    st.line_chart(data[['total_value']])
    st.subheader('Last Trades')
    st.dataframe(data.tail(10))
    st.download_button('Download CSV', data.to_csv(index=False), file_name=f'{agent}.csv')
 main
