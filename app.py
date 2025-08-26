import streamlit as st

from pages import (
    live_analytics,
    models_page,
    training_page,
    system_monitor,
    data_management,
    reports_page,
    alerts_page,
    settings_page,
)


st.set_page_config(page_title="AI Trading Dashboard", layout="wide")


with st.sidebar:
    st.title("AI Trading Dashboard")
    st.write("Manage multiple trading agents")


live_tab, models_tab, train_tab, monitor_tab, data_tab, reports_tab, alerts_tab, settings_tab = st.tabs(
    [
        "Live Analytics",
        "Models",
        "Training",
        "System Monitor",
        "Data Management",
        "Reports",
        "Alerts",
        "Settings",
    ]
)

with live_tab:
    live_analytics.render()

with models_tab:
    models_page.render()

with train_tab:
    training_page.render()

with monitor_tab:
    system_monitor.render()

with data_tab:
    data_management.render()

with reports_tab:
    reports_page.render()

with alerts_tab:
    alerts_page.render()

with settings_tab:
    settings_page.render()


