import streamlit as st


def render():
    st.header("Settings")
    st.caption("Server and user configuration")

    st.text_input("Server Host", value="localhost")
    st.number_input("Auto-Retrain Interval (hours)", value=24)
    st.text_input("Add User")
    st.selectbox("Role", ["admin", "trader", "viewer"])
    st.text_area("API Playground", "")
    st.button("Apply")

    st.markdown("---")
    st.subheader("Extensions")
    st.info("Placeholder for plugin system")

