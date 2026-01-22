import streamlit as st
from core.visualizer import get_visualization_advice


def show_dataset_preview(df):
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.markdown(
        f"""
        **Rows:** {df.shape[0]}  
        **Columns:** {df.shape[1]}
        """
    )


def show_problem_understanding(problem_info):
    st.subheader("Problem Understanding")

    st.markdown(f"**ML Type:** {problem_info['ml_type'].upper()}")
    st.markdown(f"**Task Type:** {problem_info['task_type'].capitalize()}")
    st.markdown(f"**Target Type:** {problem_info['target_type']}")

    if problem_info.get("source") == "llm":
        st.caption("🤖 AI-assisted reasoning")

    st.markdown("**Reasoning:**")
    st.write(problem_info["reasoning"])


def show_cleaning_steps(steps):
    st.subheader("Data Cleaning Guidance")

    for step in steps:
        st.markdown(f"### {step['title']}")

        if step.get("source") == "llm":
            st.caption("🤖 AI-assisted suggestion")

        st.markdown(step["reason"])
        st.code(step["code"], language="python")


def show_model_plans(plans):
    st.subheader("Model Planning")

    for plan in plans:
        st.markdown(f"### {plan['title']}")

        if plan.get("source") == "llm":
            st.caption("🤖 AI-assisted explanation")

        st.markdown(plan["reason"])
        st.code(plan["model"], language="python")


# ===============================
# ✅ NEW — Visualization section
# ===============================
def show_visualizations(df, target_col=None, session_state=None):
    st.subheader("Data Visualization")

    advice = get_visualization_advice(
        df,
        target_col=target_col,
        session_state=session_state
    )

    if advice:
        st.caption("🤖 AI-assisted visualization guidance")
        st.markdown(advice)
    else:
        st.markdown(
            "- Distribution plots to understand feature spread\n"
            "- Correlation plots to identify relationships\n"
            "- Boxplots to inspect variability"
        )
