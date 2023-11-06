import streamlit as st

import optimization
import app_options as ao
from app_manager import AppManager


def optimize(design_monitor_empty):
    with st.spinner("Optimizing structure..."):
        st.session_state.solution, st.session_state.objective_value = optimization.optimize(
            st.session_state.options,
            design_monitor_empty,
        )
    st.success("Optimization complete!")


def run():
    st.subheader("Design Options", anchor=False)
    st.session_state.options = ao.get_options_from_ui()

    tools_container, monitor_container = st.columns([2, 4])

    with tools_container:
        st.subheader("Design Tools", anchor=False)
        do_optimize = st.button("Optimize Structure", use_container_width=True)

    with monitor_container:
        st.subheader("Design Monitor", anchor=False)
        design_monitor_container = st.container()

    if do_optimize:
        with monitor_container:
            optimize(design_monitor_container)


if __name__ == "__main__":
    with AppManager():
        run()
