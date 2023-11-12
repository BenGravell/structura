import subprocess

import pyvista
import streamlit as st

import app_options as ao


def streamlit_setup():
    st.set_page_config(page_title="Structura", page_icon="🌉", layout="wide")


def check_xvfb_running():
    """Check if xvfb is already running on the machine.

    This uses the procps package, which is a requirement in packages.txt
    """
    try:
        completed_process = subprocess.run(["pgrep", "Xvfb"], capture_output=True)
        return completed_process.returncode == 0
    except Exception:
        return False


def start_xvfb():
    if not check_xvfb_running():
        # Start virtual frame buffer to avoid error & server crash with xserver not running.
        # This needed for headless deployments running on Linux servers e.g. Streamlit Community Cloud.
        try:
            pyvista.start_xvfb()
            st.toast("xvfb was not running, started it!", icon="🏳️")
        except OSError:
            # Need to handle OSError for when the app is running in non-Linux environments e.g. Windows.
            # In this case, we assume that xvfb does not actually need to be started, and we do nothing.
            pass


def first_time():
    st.session_state.options = ao.Options()
    start_xvfb()
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
