"""Structura: A Topology Optimization app."""

import streamlit as st

from app_manager import AppManager


def run():
    st.title("Structura", anchor=False)
    st.caption("Create lightweight structures with optimal stiffness.")

    cols = st.columns([2, 1])

    with cols[0]:
        st.subheader("What is Structura?", divider="blue", anchor=False)
        st.markdown(
            "Structura is an app that uses optimization techniques to design structures that handle static loads. A"
            " minimum compliance problem is posed and solved with optimality criterion-based density updates."
        )

    with cols[1]:
        st.image("assets/welcome.jpg")

    st.info("Explore the other pages in the sidebar!", icon="↩️")


if __name__ == "__main__":
    with AppManager():
        run()
