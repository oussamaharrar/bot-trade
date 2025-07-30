import os
import streamlit as st


def render():
    st.header("Data Management")
    st.caption("Browse and update datasets")

    dataset = 'training_dataset.csv'
    if os.path.exists(dataset):
        info = os.stat(dataset)
        st.write(f"Dataset size: {info.st_size/1024:.2f} KB")
        st.write(f"Last modified: {info.st_mtime}")
        st.download_button('Download', dataset)
    else:
        st.info('Dataset not found')

    st.file_uploader('Import new dataset', type=['csv'])
    st.markdown("---")
    st.subheader("Split/Merge Datasets")
    st.info("Placeholder for dataset splitting and merging")

