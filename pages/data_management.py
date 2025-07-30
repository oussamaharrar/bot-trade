import streamlit as st
from pathlib import Path
import pandas as pd

DATA_PATH = Path('training_dataset.csv')
DATASETS_DIR = Path('results')


def render():
    st.header('Data Management')
    st.caption('Manage datasets and import new data.')

    if DATA_PATH.exists():
        st.subheader('Training Dataset')
        df = pd.read_csv(DATA_PATH).head(10)
        st.dataframe(df)
        st.write(f'Size: {DATA_PATH.stat().st_size/1024:.2f} KB')
    else:
        st.info('No training dataset found.')

    st.markdown('### Result Files')
    files = list(DATASETS_DIR.glob('run_*.csv'))
    for f in files:
        st.write(f.name, f'({f.stat().st_size/1024:.1f} KB)')
        with st.expander('Preview'):
            st.dataframe(pd.read_csv(f).head())
        if st.button('Delete', key=f'del_{f.name}'):
            f.unlink()
            st.experimental_rerun()

    st.file_uploader('Import CSV', type=['csv'])
