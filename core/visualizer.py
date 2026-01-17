import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_numeric_columns(df):
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


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
        ax.set_title(f"Outlier Check: {col}")
        figures.append(fig)

    return figures
