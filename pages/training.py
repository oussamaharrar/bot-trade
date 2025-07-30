import streamlit as st
import subprocess

DATA_FILE = 'training_dataset.csv'


def render():
    st.header('Training Center')
    st.caption('Configure training parameters and launch training jobs.')

    with st.form('train_params'):
        learning_rate = st.number_input('Learning rate', 0.0001, 1.0, 0.001, help='Optimizer learning rate')
        batch_size = st.number_input('Batch size', 1, 1024, 32, help='Training batch size')
        steps = st.number_input('Training steps', 1, 100000, 1000, help='Total training steps')
        policy = st.selectbox('Policy', ['MlpPolicy', 'CnnPolicy'], help='RL policy architecture')
        dataset_slice = st.text_input('Dataset slice (e.g., symbol or period)', help='Subset of data to use')
        start = st.form_submit_button('Start Training')
    if start:
        st.session_state.training_process = subprocess.Popen(['python', 'train_rl.py'])
        st.success('Training started in background')

    if 'training_process' in st.session_state:
        proc = st.session_state.training_process
        if proc.poll() is None:
            st.info('Training running...')
        else:
            st.success('Training finished')
            del st.session_state['training_process']
