import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from llm_engine.llm_client import call_llm


def get_numeric_columns(df):
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


# ===============================
# LLM advisory (prioritization + explanation)
# ===============================
def get_visualization_advice(df, target_col=None, max_cols=8, session_state=None):
    """
    LLM-assisted visualization prioritization.
    Advisory only. No plotting, no execution.
    """
    try:
        cols = df.columns.tolist()[:max_cols]
        dtypes = {c: str(df[c].dtype) for c in cols}

        prompt = f"""
You are a senior data scientist.

Dataset summary:
- Columns: {cols}
- Dtypes: {dtypes}
- Target: {target_col}

Task:
Suggest the most important visualizations to understand this dataset
and briefly explain why each is useful.

Rules:
- Do NOT generate code
- Do NOT execute anything
- Max 5 bullet points
- Plain text only
"""

        return call_llm(
            prompt=prompt,
            cache_key="viz_advice",
            session_state=session_state
        )

    except Exception:
        return None


# ===============================
# RULE-BASED PLOTS (UNCHANGED)
# ===============================
def plot_target_distribution(df, target_col):
    fig, ax = plt.subplots()
    df[target_col].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Target Distribution")
    return fig


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if numeric_df.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def plot_boxplots(df):
    numeric_cols = get_numeric_columns(df)
    figures = []

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Distribution & Outlier Check: {col}")
        figures.append(fig)

    return figures
