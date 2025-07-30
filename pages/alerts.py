import streamlit as st


def render():
    st.header('Alerts & Notifications')
    st.caption('Configure push notifications and alert rules.')

    st.checkbox('Enable Telegram alerts')
    st.checkbox('Enable Discord alerts')
    st.checkbox('Enable SMS alerts')
    st.text_input('Telegram token', type='password')
    st.text_input('Discord webhook URL')
    st.text_input('Phone number')
    st.text_area('Alert rules', help='Define when to send alerts')
    st.button('Save Settings')
