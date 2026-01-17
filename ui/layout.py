import streamlit as st
from ui.style import apply_global_style


def render_header():
    apply_global_style()

    st.markdown(
        """
        <h1 style="text-align:center;">
            AI-Guided Machine Learning & Deep Learning Project Mentor
        </h1>
        <p style="text-align:center; font-size:16px;">
            A step-by-step mentor that thinks like a senior ML engineer.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()
