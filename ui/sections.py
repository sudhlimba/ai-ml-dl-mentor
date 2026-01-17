import streamlit as st


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
    st.markdown("**Reasoning:**")
    st.write(problem_info["reasoning"])


def show_cleaning_steps(steps):
    st.subheader("Data Cleaning Guidance")
    for step in steps:
        st.markdown(f"### {step['title']}")
        st.markdown(step["reason"])
        st.code(step["code"], language="python")


def show_model_plans(plans):
    st.subheader("Model Planning")
    for plan in plans:
        st.markdown(f"### {plan['title']}")
        st.markdown(plan["reason"])
        st.code(plan["model"])
