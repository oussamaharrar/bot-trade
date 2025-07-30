import os
import streamlit as st
from utils import list_reports


def render():
    st.header("Reports")
    st.caption("Generated PDF and CSV reports")

    reports = list_reports()
    if reports:
        for path in reports:
            st.download_button(os.path.basename(path), path)
    else:
        st.info("No reports available")


