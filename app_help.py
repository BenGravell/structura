import streamlit as st


def help():
    st.header("Problem Setup", anchor=False)
    st.markdown(
        "The problem setup is the symmetric Messerschmitt-Bölkow-Blohm (MBB) beam, which represents half of a simply"
        " supported beam with a center point load. The full beam is cut in half and the cut is replaced with a boundary"
        " condition with horizontal restraint and vertical translation free."
    )
    st.image(
        "assets/smbb_domain.png",
        caption=(
            "The design domain, boundary conditions, and external load for the optimization of a symmetric MBB beam."
        ),
    )
