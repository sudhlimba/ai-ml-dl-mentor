import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from core.data_loader import load_csv, get_basic_info
from core.data_profiler import profile_dataset
from core.visualizer import plot_correlation_heatmap, plot_boxplots
from core.cleaning_guide import get_cleaning_guidance
from core.model_planner import plan_models
from core.train_test_guide import get_train_test_guidance
from core.auto_cleaner import auto_clean_dataframe
from core.custom_visualizer import generate_custom_plot

from llm_engine.prompts import problem_understanding_prompt
from llm_engine.llm_client import call_llm
from llm_engine.response_parser import parse_llm_response

from ui.style import apply_global_style

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("df", None)
st.session_state.setdefault("problem_info", None)
st.session_state.setdefault("fullscreen_fig", None)
st.session_state.setdefault("fullscreen_title", None)

# ===============================
# HELPERS
# ===============================
def rotate_axis_labels(fig):
    for ax in fig.axes:
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()


def render_small(fig, width=420):
    rotate_axis_labels(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, width=width)


def render_fullscreen(fig):
    rotate_axis_labels(fig)
    fig.set_size_inches(12, 9)
    st.pyplot(fig)
    if st.button("❌ Close Fullscreen"):
        st.session_state.fullscreen_fig = None
        st.experimental_rerun()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI-Guided ML Mentor", layout="wide")
apply_global_style()

# ===============================
# HEADER
# ===============================
st.markdown("<h1>AI-Guided Machine Learning & Deep Learning Project Mentor</h1>", unsafe_allow_html=True)
st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

# ===============================
# FULLSCREEN MODE
# ===============================
if st.session_state.fullscreen_fig is not None:
    st.markdown("## 🔍 Fullscreen Graph View")
    render_fullscreen(st.session_state.fullscreen_fig)
    st.stop()

# ===============================
# DATASET UPLOAD
# ===============================
st.markdown("## Dataset Upload")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.session_state.df = load_csv(uploaded_file)
    st.dataframe(st.session_state.df.head())

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

# ===============================
# PROJECT GOAL
# ===============================
if st.session_state.df is not None:
    st.markdown("## Project Goal")
    goal = st.text_input("Describe what you want to build")

    if goal:
        info = get_basic_info(st.session_state.df)
        raw = call_llm(problem_understanding_prompt(goal, info), fallback_context=goal)
        st.session_state.problem_info = parse_llm_response(raw)
        st.write(st.session_state.problem_info["reasoning"])

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

# ===============================
# DATA VISUALIZATION
# ===============================
if st.session_state.df is not None:
    st.markdown("## Data Visualization")

    visuals = []

    heatmap = plot_correlation_heatmap(st.session_state.df)
    if heatmap:
        visuals.append((
            "Correlation Heatmap",
            "sns.heatmap(df.corr(), annot=True)",
            "Axes are features; colors show correlation strength.",
            "Strong correlations indicate important predictors.",
            heatmap
        ))

    for fig in plot_boxplots(st.session_state.df):
        visuals.append((
            "Boxplot",
            "sns.boxplot(x=df[column])",
            "Shows distribution and outliers.",
            "Detects invalid values like BP = 0.",
            fig
        ))

    if visuals:
        idx = st.slider("Browse visualizations", 0, len(visuals) - 1, 0)
        title, code, how, learn, fig = visuals[idx]

        st.markdown("<div class='ml-card'>", unsafe_allow_html=True)
        left, right = st.columns([1, 1])

        with left:
            st.markdown(f"### {title}")
            st.code(code)
            st.markdown("**How to read**")
            st.write(how)
            st.markdown("**What we learn**")
            st.write(learn)

            if st.button("🔍 View Fullscreen", key=f"fs_auto_{idx}"):
                st.session_state.fullscreen_fig = fig
                st.experimental_rerun()

        with right:
            render_small(fig)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

# ===============================
# BUILD YOUR OWN GRAPH
# ===============================
if st.session_state.df is not None:
    st.markdown("## Build Your Own Graph")

    plot_type = st.selectbox(
        "Graph type",
        ["Histogram", "Boxplot", "Scatter Plot", "Line Plot", "Count Plot", "Correlation Heatmap"]
    )

    cols = list(st.session_state.df.columns)
    features = []

    if plot_type in ["Histogram", "Boxplot", "Count Plot"]:
        features = [st.selectbox("Feature", cols)]
    elif plot_type in ["Scatter Plot", "Line Plot"]:
        features = [st.selectbox("X-axis", cols), st.selectbox("Y-axis", cols)]

    fig = generate_custom_plot(st.session_state.df, plot_type, features)

    st.markdown("<div class='ml-card'>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"### {plot_type}")
        st.code(f"generate_custom_plot(df, '{plot_type}', {features})")
        st.markdown("**How to read**")
        st.write("Interpret axes based on selected features.")
        st.markdown("**What we learn**")
        st.write("Custom graphs help validate assumptions.")

        if st.button("🔍 View Fullscreen", key="fs_custom"):
            st.session_state.fullscreen_fig = fig
            st.experimental_rerun()

    with right:
        render_small(fig)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

# ===============================
# REMAINING SECTIONS (UNCHANGED)
# ===============================
if st.session_state.df is not None:
    st.markdown("## Dataset Health Report")
    st.dataframe(profile_dataset(st.session_state.df))

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

if st.session_state.df is not None:
    st.markdown("## Data Cleaning")
    auto = st.checkbox("Apply automatic cleaning")

    if auto:
        cleaned = auto_clean_dataframe(st.session_state.df)
        st.download_button("Download cleaned CSV", cleaned.to_csv(index=False), "cleaned.csv")
    else:
        for step in get_cleaning_guidance(st.session_state.df):
            st.markdown(f"### {step['title']}")
            st.write(step["reason"])
            st.code(step["code"])

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

if st.session_state.problem_info is not None:
    st.markdown("## Train-Test Split Strategy")
    for s in get_train_test_guidance(st.session_state.problem_info["task_type"]):
        st.markdown(f"### {s['title']}")
        st.write(s["why"])
        st.code(s["code"])

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

if st.session_state.problem_info is not None:
    st.markdown("## Model Selection")
    for p in plan_models(st.session_state.problem_info):
        st.markdown(f"### {p['title']}")
        st.write(p["reason"])
        st.code(p["model"])
