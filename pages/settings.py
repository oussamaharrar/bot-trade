import streamlit as st
import yaml

CONFIG_PATH = 'config.yaml'


def render():
    st.header('Settings')
    st.caption('Modify server parameters and integrations.')

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if 'config_edit' not in st.session_state:
        st.session_state.config_edit = yaml.dump(config)

    st.text_area('Config YAML', st.session_state.config_edit, key='config_edit', height=300)
    if st.button('Save Config'):
        with open(CONFIG_PATH, 'w') as f:
            f.write(st.session_state.config_edit)
        st.success('Configuration saved')
