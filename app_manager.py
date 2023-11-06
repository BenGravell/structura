import streamlit as st

import app_options as ao


def streamlit_setup():
    st.set_page_config(page_title="Structura", page_icon="🌉", layout="wide")


def first_time():
    st.session_state.options = ao.Options()
    st.session_state.initialized = True


def finalize():
    pass


class AppManager:
    def __enter__(self):
        streamlit_setup()
        if not st.session_state.get("initialized"):
            first_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        finalize()
        return False
