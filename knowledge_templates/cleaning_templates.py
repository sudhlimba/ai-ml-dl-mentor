def get_cleaning_explanations():
    """
    Standardized teaching templates for data cleaning.
    """

    return [
        {
            "title": "Missing Values",
            "why": "Missing data can confuse models and reduce reliability.",
            "best_practice": "Use median for numeric and mode for categorical values."
        },
        {
            "title": "Outliers",
            "why": "Outliers can bias training and distort patterns.",
            "best_practice": "Detect using IQR or Z-score methods."
        },
        {
            "title": "Categorical Encoding",
            "why": "Models work only with numbers.",
            "best_practice": "Use one-hot encoding for nominal categories."
        },
        {
            "title": "Feature Scaling",
            "why": "Different feature scales can dominate learning.",
            "best_practice": "Use StandardScaler or MinMaxScaler."
        }
    ]
