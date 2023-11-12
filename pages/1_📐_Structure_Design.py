import streamlit as st

import optimization
import app_options as ao
from app_manager import AppManager
from ui.design_monitor import DesignMonitor


def optimize(design_monitor):
    with st.spinner("Optimizing structure..."):
        st.session_state.solution, st.session_state.solution_for_display, st.session_state.objective_value = (
            optimization.optimize(
                st.session_state.options,
                st.session_state.get("solution"),
                design_monitor,
            )
        )
    st.success("Optimization complete!")


def randomize(design_monitor):
    with st.spinner("Randomizing structure..."):
        st.session_state.solution, st.session_state.solution_for_display, st.session_state.objective_value = (
            optimization.randomize(
                st.session_state.options,
                st.session_state.get("solution"),
                design_monitor,
            )
        )
    st.success("Randomization complete!")


def run():
    st.subheader("Design Options", anchor=False)
    st.session_state.options = ao.get_options_from_ui()

    tools_container, monitor_container = st.columns([2, 4])

    with tools_container:
        st.subheader("Design Tools", anchor=False)
        do_optimize = st.button("Optimize Structure", use_container_width=True)
        do_randomize = st.button("Randomize Structure", use_container_width=True)

    with monitor_container:
        st.subheader("Design Monitor", anchor=False)
        design_monitor = DesignMonitor()

    with tools_container:
        if do_optimize:
            optimize(design_monitor)

        if do_randomize:
            randomize(design_monitor)


if __name__ == "__main__":
    with AppManager():
        run()
