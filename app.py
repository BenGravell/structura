"""Structura: A Topology Optimization app."""

import streamlit as st
from stpyvista import stpyvista


import utils
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

            show_3d_model = st.toggle("Show 3D Model")
            if show_3d_model:
                st.markdown("""Controls: 
                            - LMB + Drag: Free Rotate
                            - Ctrl or Alt + LMB + Drag: Rotate about Center
                            - Shift + LMB + Drag: Pan
                            - Scroll: Zoom""")
                with st.spinner("Rendering 3D model..."):
                    x, y, z = utils.get_pv_xyz_data(st.session_state.solution, st.session_state.options.mirror)
                    plotter = utils.get_pv_plotter(x, y, z)
                stpyvista(plotter, horizontal_align="center", panel_kwargs={"orientation_widget": True})

    with tabs[tab_names.index("Help")]:
        app_help.help()


if __name__ == "__main__":
    main()
