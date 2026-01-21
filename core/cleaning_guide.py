def get_cleaning_guidance(df, llm_suggestions=None):
    """
    Returns step-by-step data cleaning guidance.
    Rule-based by default.
    Optionally enriched with LLM suggestions (advisory only).
    Safe against None inputs.
    """

    if df is None:
        return []

    steps = []

    def enrich(step):
        """
        Attach LLM suggestion if available.
        """
        if llm_suggestions:
            step["reason"] += f"\n\nLLM Suggestion:\n{llm_suggestions}"
            step["source"] = "llm"
        return step

    # ===============================
    # Missing Values
    # ===============================
    if df.isnull().sum().sum() > 0:
        steps.append(enrich({
            "title": "Handling Missing Values",
            "reason": (
                "Missing values can confuse models, reduce accuracy, "
                "and introduce bias if not handled carefully."
            ),
            "code": """
import pandas as pd

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=["object"]).columns
df[categorical_cols] = df[categorical_cols].fillna(
    df[categorical_cols].mode().iloc[0]
)
"""
        }))

    # ===============================
    # Outliers
    # ===============================
    steps.append(enrich({
        "title": "Handling Outliers",
        "reason": (
            "Outliers can heavily influence model learning and distort patterns, "
            "especially in medical and financial datasets."
        ),
        "code": """
import numpy as np

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
"""
    }))

    # ===============================
    # Encoding
    # ===============================
    steps.append(enrich({
        "title": "Encoding Categorical Features",
        "reason": (
            "Machine learning models only understand numbers. "
            "Categorical variables must be converted into numeric form."
        ),
        "code": """
import pandas as pd

df = pd.get_dummies(df, drop_first=True)
"""
    }))

    # ===============================
    # Scaling
    # ===============================
    steps.append(enrich({
        "title": "Feature Scaling",
        "reason": (
            "Scaling ensures all numeric features contribute equally, "
            "especially important for distance-based models."
        ),
        "code": """
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
"""
    }))

    return steps
