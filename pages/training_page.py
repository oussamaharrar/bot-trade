import streamlit as st
import subprocess


def render():
    st.header("Training Center")
    st.caption("Configure and launch agent training")

    with st.form("train_form"):
        col1, col2 = st.columns(2)
        with col1:
            lr = st.number_input("Learning Rate", value=0.0003, format="%f", help="Optimizer learning rate")
            batch = st.number_input("Batch Size", value=32, help="Training batch size")
            steps = st.number_input("Steps", value=10000, help="Total training steps")
        with col2:
            dataset = st.text_input("Dataset Path", value="training_dataset.csv")
            policy = st.selectbox("Policy", ["MlpPolicy", "CnnPolicy"], help="RL policy type")
            reward = st.text_input("Reward shaping", value="default")
        start_btn = st.form_submit_button("Start Training")
        if start_btn:
            st.session_state['training_status'] = 'running'
            subprocess.Popen(["python", "train_rl.py"])  # placeholder
            st.success("Training started in background")

    status = st.session_state.get('training_status')
    if status:
        st.info(f"Training status: {status}")
        st.progress(0)

    st.markdown("---")
    st.subheader("Error/Log Analyzer")
    st.info("Placeholder for log analysis and suggestions")

