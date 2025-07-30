import streamlit as st
from pathlib import Path

REPORTS_DIR = Path('reports')


def render():
    st.header('Reports')
    st.caption('Browse and download generated reports.')

    REPORTS_DIR.mkdir(exist_ok=True)
    reports = list(REPORTS_DIR.glob('*.*'))
    if not reports:
        st.info('No reports found.')
        return

    for rep in reports:
        cols = st.columns([4,1])
        cols[0].write(rep.name)
        cols[1].download_button('Download', rep.read_bytes(), file_name=rep.name)
