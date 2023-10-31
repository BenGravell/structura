"""Structura: A Topology Optimization app."""

import streamlit as st

import welcome
import optimization
import app_options as ao
import app_help



def main():
    st.set_page_config(page_title="Structura", page_icon="🌉", layout="centered")

    tab_names = [
        "Welcome",
        "Options",
        "Results",
        "Help",
    ]
    tabs = st.tabs(tab_names)

    with tabs[tab_names.index("Welcome")]:
        welcome.run()

    with tabs[tab_names.index("Options")]:
        st.session_state.options = ao.get_options_from_ui()

    with tabs[tab_names.index("Results")]:
        button_container = st.container()

        # Create a placeholder to hold the animated structure image
        if st.session_state.get("image_container") is None:
            st.session_state.image_container = st.empty()

        with button_container:
            do_optimize = st.button("Optimize Structure")
            if do_optimize:
                with st.spinner("Optimizing structure..."):
                    st.session_state.solution_frame = optimization.optimize(st.session_state.options, st.session_state.image_container)

        if st.session_state.get("solution_frame") is not None:
            st.session_state.image_container.image(
                st.session_state.solution_frame, caption="Solution", use_column_width=True
            )

    with tabs[tab_names.index("Help")]:
        app_help.help()


if __name__ == "__main__":
    main()
