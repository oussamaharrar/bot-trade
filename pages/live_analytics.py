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


