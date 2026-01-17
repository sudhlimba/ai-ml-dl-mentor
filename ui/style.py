import streamlit as st


def apply_global_style():
    st.markdown(
        """
        <style>
        /* ===== GLOBAL BACKGROUND ===== */
        html, body, [class*="css"] {
            background-color: #222831;
            color: #ffffff;
        }

        /* ===== HEADINGS ===== */
        h1, h2, h3, h4, h5 {
            color: #ffffff;
        }

        /* ===== FLOATING CARD ===== */
        .ml-card {
            max-width: 900px;
            margin: 0 auto 50px auto;
            padding: 24px;
            background-color: #393E46;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        }

        /* ===== SECTION SPACING ===== */
        .section-space {
            height: 70px;
        }

        /* ===== ACCENT TEXT (OPTIONAL) ===== */
        .accent {
            color: #948979;
        }

        /* ===== STREAMLIT WIDGET FIXES ===== */
        label, .stTextInput, .stSelectbox, .stCheckbox {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
