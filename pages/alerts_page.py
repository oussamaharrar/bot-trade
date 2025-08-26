import streamlit as st


def render():
    st.header("Alerts & Notifications")
    st.caption("Configure push notifications")

    st.text_input("Telegram Token")
    st.text_input("Discord Webhook")
    st.text_input("SMS API Key")
    st.checkbox("Trade Signal Alert")
    st.checkbox("Error Alert")
    st.checkbox("Training Done Alert")
    st.button("Save Settings")


