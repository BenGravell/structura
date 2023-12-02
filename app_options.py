from dataclasses import dataclass, astuple
from enum import Enum
from typing import Callable

import matplotlib.cm as cm
import streamlit as st

import constants


class FilterType(Enum):
    NO_FILTER = 0
    DENSITY_FILTER = 1
    SENSITIVITY_FILTER = 2

    @property
    def description(self):
        return FILTER_MAP[self.value]


FILTER_MAP = {
    FilterType.NO_FILTER: "No Filter",
    FilterType.DENSITY_FILTER: "Density Filter",
    FilterType.SENSITIVITY_FILTER: "Sensitivity Filter",
}


@dataclass
class Options:
    nel: int = 2000
    volfrac: float = 0.3
    rmin: float = 1.5
    penal: float = 2.0
    filter_type: int = 1
    density_initialization_method: str = "random"
    move: float = 0.2
    change_tol: float = 0.01
    max_iters: int = 100
    cmap: Callable | None = None
    upscale_factor: int = 4
    upscale_method: str = "NEAREST"
    mirror: bool = True

    def unpack(self):
        return list(astuple(self))


def options_update_button_callback():
    st.toast("Updated Options!", icon="🍱")


def get_options_from_ui() -> Options:
    with st.form("options_form"):
        st.subheader("Design Options", anchor=False)

        form_button_container = st.container()

        options_categories = [
            "Simulation Domain",
            "Objective",
            "Constraints",
            "Solver",
            "Display",
        ]

        tabs = st.tabs(options_categories)

        with tabs[options_categories.index("Simulation Domain")]:
            nel = st.select_slider(
                "Target Total Number of Elements",
                options=[200, 500, 1000, 2000, 5000, 10000, 20000],
                value=2000,
            )
            density_initialization_method = st.selectbox(
                "Density Initialization Method",
                options=["continue", "random", "constant"],
                help=(
                    'The option "continue" will continue from the existing solution if available, otherwise will use'
                    ' "random".'
                ),
            )

        with tabs[options_categories.index("Constraints")]:
            volfrac = st.slider(
                "Volume Fraction",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=0.30,
            )

        with tabs[options_categories.index("Objective")]:
            filter_type = st.selectbox(
                "Filter Type",
                options=[i for i in FILTER_MAP],
                format_func=FILTER_MAP.get,
                index=1,
            )
            rmin = st.slider(
                "Filter Radius",
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                value=1.5,
            )
            penal = st.slider(
                "Penalty Power",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help=(
                    "Power to which to raise density design variables. Higher values discourage intermediate values"
                    " more heavily, but tend to lock the optimizer into local minima with fine structural details that"
                    " are globally suboptimal."
                ),
            )

        with tabs[options_categories.index("Solver")]:
            move = st.select_slider(
                "Design Variable Change Max per Iteration",
                options=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                value=0.2,
            )
            change_tol = st.select_slider(
                "Design Variable Change Convergence Tolerance",
                options=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                value=0.01,
            )
            max_iters = st.select_slider(
                "Max Iterations",
                options=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
                value=100,
            )

        with tabs[options_categories.index("Display")]:
            mirror = st.toggle("Show Full Beam by Mirroring Half Beam", value=True)

            cmap_name = st.selectbox(
                "Colormap", constants.COLORMAP_OPTIONS, index=constants.COLORMAP_OPTIONS.index("Blues")
            )
            cmap = cm.get_cmap(cmap_name)

            upscale_factor = st.select_slider("Upscale Factor", options=[1, 2, 4, 8, 16, 32, 64], value=4)
            upscale_method = st.selectbox("Upscaling Method", options=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"])

        with form_button_container:
            st.form_submit_button("Update Options", on_click=options_update_button_callback)

    return Options(
        nel,
        volfrac,
        rmin,
        penal,
        filter_type,
        density_initialization_method,
        move,
        change_tol,
        max_iters,
        cmap,
        upscale_factor,
        upscale_method,
        mirror,
    )
