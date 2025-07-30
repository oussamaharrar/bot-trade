import streamlit as st
from pathlib import Path

# Import page modules
from pages import (
    models,
    training,
    live_analytics,
    system_monitor,
    data_management,
    reports,
    alerts,
    settings,
)

st.set_page_config(
    page_title='AI Trading Dashboard',
    layout='wide',
    page_icon=':chart_with_upwards_trend:',
    initial_sidebar_state='expanded'
)

st.sidebar.title('Agents')
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

agents = [p.name for p in models_dir.glob('*.pkl')]
if not agents:
    st.sidebar.write('No agents available')
else:
    for ag in agents:
        st.sidebar.write(ag)

st.sidebar.markdown('---')
st.sidebar.write('Global Actions')
if st.sidebar.button('Run Bot', help='Execute trading bot'):
    st.sidebar.success('Bot started (placeholder)')


# Main tabs
live_tab, models_tab, train_tab, monitor_tab, data_tab, reports_tab, alerts_tab, settings_tab = st.tabs(
    [
        'Live Analytics', 'Models', 'Training', 'System Monitor',
        'Data Management', 'Reports', 'Alerts', 'Settings'
    ]
)

with live_tab:
    live_analytics.render()

with models_tab:
    models.render()

with train_tab:
    training.render()

with monitor_tab:
    system_monitor.render()

with data_tab:
    data_management.render()

with reports_tab:
    reports.render()

with alerts_tab:
    alerts.render()

with settings_tab:
    settings.render()
