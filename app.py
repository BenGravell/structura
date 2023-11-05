"""Structura: A Topology Optimization app."""

import pandas as pd
import streamlit as st

import utils
import welcome
import optimization
import app_options as ao
import app_help


def main():
    st.set_page_config(page_title="Structura", page_icon="🌉", layout="wide")

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
                    st.session_state.solution, st.session_state.objective_value = optimization.optimize(
                        st.session_state.options, st.session_state.image_container
                    )

        if st.session_state.get("solution") is not None:
            solution_frame = utils.x2frame(
                st.session_state.solution,
                st.session_state.options.cmap,
                st.session_state.options.upscale_factor,
                st.session_state.options.upscale_method,
                st.session_state.options.mirror,
            )
            st.image(
                solution_frame,
                caption=f"Solution, Objective = {st.session_state.objective_value:12.3f}",
                use_column_width=True,
            )

            if st.session_state.get("plotter") is None:
                st.session_state.plotter = None

            do_render_3d_model = st.button("Render 3D Model")
            if do_render_3d_model:
                with st.spinner("Rendering 3D model..."):
                    st.session_state.plotter = utils.get_pv_plotter(
                        st.session_state.solution, st.session_state.options.mirror, st.session_state.options.cmap
                    )
            try:
                from stpyvista import stpyvista
            except ImportError as exc:
                st.exception(exc)

            st.caption("PyVista 3D Viewer Controls")
            pyvista_3d_viewer_controls_df = pd.DataFrame.from_dict(
                {
                    "Control": ["LMB + Drag", "Ctrl or Alt + LMB + Drag", "Shift + LMB + Drag", "Scroll"],
                    "Description": ["Free Rotate", "Rotate about Center", "Pan", "Zoom"],
                }
            ).set_index("Control")
            st.dataframe(pyvista_3d_viewer_controls_df)

            plotter = st.session_state.get("plotter")
            if plotter is not None:
                stpyvista(st.session_state.plotter, horizontal_align="left", panel_kwargs={"orientation_widget": True})

    with tabs[tab_names.index("Help")]:
        app_help.help()


if __name__ == "__main__":
    main()
