import pandas as pd
import streamlit as st

import utils
from app_manager import AppManager


def run():
    if st.session_state.get("solution") is not None:
        solution_frame = utils.x2frame(
            st.session_state.solution,
            st.session_state.options.cmap,
            st.session_state.options.upscale_factor,
            st.session_state.options.upscale_method,
            st.session_state.options.mirror,
        )

        st.header("2D Image", anchor=False)
        st.image(
            solution_frame,
            caption=f"Solution, Objective = {st.session_state.objective_value:12.3f}",
            use_column_width=True,
        )

        st.header("3D Model", anchor=False)
        if st.session_state.get("plotter") is None:
            st.session_state.plotter = None

        cols = st.columns(2)

        with cols[0]:
            st.subheader("3D Model Options", anchor=False)
            with st.form("3d_model_options"):
                use_cmap = st.toggle(
                    "Use Colormap",
                    value=True,
                    help=(
                        "If disabled, the 3D model will just be a solid color. If enabled, the 3D model will be colored"
                        " by thickness with the same colormap as in the 2D image."
                    ),
                )
                show_edges = st.toggle("Show Edges")
                thickness_resolution = st.select_slider(
                    "Thickness Resolution",
                    options=[2, 5, 10, 20, 50, 100],
                    value=10,
                    help=(
                        "Number of levels in the thickness for voxelization. Higher numbers will yield a more refined"
                        " voxelization, but will take longer to render."
                    ),
                )
                st.form_submit_button("Update 3D Model Options")

        with cols[1]:
            st.subheader("3D Viewer Controls", anchor=False)
            pyvista_3d_viewer_controls_df = pd.DataFrame.from_dict(
                {
                    "Control": ["LMB + Drag", "Ctrl or Alt + LMB + Drag", "Shift + LMB + Drag", "Scroll"],
                    "Description": ["Free Rotate", "Rotate about Center", "Pan", "Zoom"],
                }
            ).set_index("Control")
            st.dataframe(pyvista_3d_viewer_controls_df, use_container_width=True)

        import_ok = False
        try:
            import pyvista  # noqa
            import stpyvista  # noqa

            import_ok = True
        except ImportError as exc:
            with st.expander("`pyvista` import error", expanded=False):
                st.info(
                    "There is a known issue with Streamlit Community Cloud that prevents certain package dependencies"
                    " required for `pyvista` from being imported correctly. See"
                    " https://docs.streamlit.io/knowledge-base/dependencies/libgl. Fixes & workarounds are being"
                    " explored. Try running the app locally in the meantime."
                )
                st.write(
                    "Failed to import `pyvista` or `stypvista` with the following exception. 3D model cannot be"
                    " rendered."
                )
                st.exception(exc)

        if import_ok:
            do_render_3d_model = st.button("Render 3D Model")
            if do_render_3d_model:
                with st.spinner("Rendering 3D model..."):
                    cmap_for_3d_model = st.session_state.options.cmap if use_cmap else None

                    st.session_state.plotter = utils.get_pv_plotter(
                        st.session_state.solution,
                        st.session_state.options.mirror,
                        cmap_for_3d_model,
                        show_edges,
                        thickness_resolution,
                    )

            if (plotter := st.session_state.get("plotter")) is not None:
                stpyvista.stpyvista(plotter, horizontal_align="left", panel_kwargs={"orientation_widget": True})
    else:
        st.info("No structure design found. Try using the tools on the Structure Design page to create a design!")


if __name__ == "__main__":
    with AppManager():
        run()
