import os
import subprocess
import streamlit as st

from utils import list_models


def render():
    st.header("Models & Agents")
    st.caption("Manage training agents and models")

    models = list_models()
    if not models:
        st.info("No models or agents found.")
    else:
        for m in models:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.write(m)
            col2.button("Train", key=f"train_{m}")
            col3.button("Resume", key=f"resume_{m}")
            col4.button("Pause", key=f"pause_{m}")
            col5.button("Delete", key=f"delete_{m}")

    st.subheader("Add New Agent")
    with st.form("new_agent_form"):
        name = st.text_input("Agent name")
        lr = st.number_input("Learning rate", value=0.0003, format="%f")
        batch = st.number_input("Batch size", value=32)
        submit = st.form_submit_button("Create")
        if submit:
            st.success(f"Agent '{name}' created (placeholder)")

    st.markdown("---")
    st.subheader("Multi-Agent Batch Training")
    st.info("Placeholder for launching multiple agents simultaneously.")


