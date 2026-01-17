import matplotlib.pyplot as plt
import seaborn as sns
def generate_custom_plot(df, plot_type, features):
    fig, ax = plt.subplots(figsize=(5, 3))
    if plot_type == "Histogram":
        sns.histplot(df[features[0]], kde=True, ax=ax)
    elif plot_type == "Boxplot":
        sns.boxplot(x=df[features[0]], ax=ax)
    elif plot_type == "Scatter Plot":
        sns.scatterplot(x=df[features[0]], y=df[features[1]], ax=ax)
    elif plot_type == "Line Plot":
        sns.lineplot(x=df[features[0]], y=df[features[1]], ax=ax)
    elif plot_type == "Count Plot":
        sns.countplot(x=df[features[0]], ax=ax)
    elif plot_type == "Correlation Heatmap":
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(plot_type)
    return fig
