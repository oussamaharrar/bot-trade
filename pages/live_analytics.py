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
