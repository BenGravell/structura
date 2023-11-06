import streamlit as st

from app_manager import AppManager


def run():
    st.header("Problem Setup", anchor=False)
    cols = st.columns([2, 2])
    with cols[0]:
        with open("help/problem_setup.md", mode="r", encoding="utf-8") as f:
            st.markdown(f.read())
    with cols[1]:
        st.image(
            "assets/smbb_domain.png",
            caption=(
                "The design domain, boundary conditions, and external load for the optimization of a symmetric MBB"
                " beam."
            ),
            use_column_width=True,
        )


if __name__ == "__main__":
    with AppManager():
        run()
