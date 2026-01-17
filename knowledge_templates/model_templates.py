def get_model_explanations(task_type):
    """
    Teaching templates for model planning.
    """

    if task_type == "classification":
        return [
            {
                "model": "Logistic Regression",
                "why": "Simple, fast, and easy to interpret.",
                "when": "Use as a baseline for binary or multiclass classification."
            },
            {
                "model": "Random Forest",
                "why": "Handles non-linear relationships and interactions well.",
                "when": "Use when accuracy is more important than interpretability."
            }
        ]

    if task_type == "regression":
        return [
            {
                "model": "Linear Regression",
                "why": "Simple and interpretable baseline.",
                "when": "Use when relationship is roughly linear."
            },
            {
                "model": "XGBoost",
                "why": "Powerful for structured/tabular data.",
                "when": "Use when performance matters."
            }
        ]

    return [
        {
            "model": "Neural Networks",
            "why": "Capable of learning complex patterns.",
            "when": "Use only when data size and complexity justify it."
        }
    ]
