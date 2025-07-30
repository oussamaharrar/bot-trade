import streamlit as st
from pathlib import Path
import subprocess

MODELS_DIR = Path('models')


def render():
    """Models/Agents management UI"""
    st.header('Models & Agents')
    st.caption('Manage your trading agents and monitor their status.')

    MODELS_DIR.mkdir(exist_ok=True)
    models = sorted(MODELS_DIR.glob('*.pkl'))

    # Initialize session state for agent statuses
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {m.name: 'Idle' for m in models}
    else:
        for m in models:
            st.session_state.agent_status.setdefault(m.name, 'Idle')

    for model in models:
        cols = st.columns([3, 1, 1, 1, 1, 1])
        cols[0].markdown(f"**{model.name}**")
        cols[1].write(st.session_state.agent_status.get(model.name, 'Idle'))
        if cols[2].button('Train', key=f'train_{model.name}', help='Train this model'):
            st.session_state.agent_status[model.name] = 'Training'
            subprocess.Popen(['python', 'train_model.py'])
        if cols[3].button('Resume', key=f'resume_{model.name}', help='Resume training/operation'):
            st.session_state.agent_status[model.name] = 'Active'
        if cols[4].button('Pause', key=f'pause_{model.name}', help='Pause this agent'):
            st.session_state.agent_status[model.name] = 'Paused'
        if cols[5].button('Delete', key=f'del_{model.name}', help='Delete this model'):
            model.unlink()
            st.session_state.agent_status.pop(model.name, None)
            st.experimental_rerun()

    st.markdown('---')
    st.subheader('Add New Agent')
    with st.form('add_agent'):
        name = st.text_input('Model name', help='File name for the new model')
        learning_rate = st.number_input('Learning rate', 0.0001, 1.0, 0.001)
        batch_size = st.number_input('Batch size', 1, 1024, 32)
        submitted = st.form_submit_button('Create')
        if submitted:
            (MODELS_DIR / f"{name}.pkl").touch()
            st.session_state.agent_status[f"{name}.pkl"] = 'Idle'
            st.success('Agent added')
