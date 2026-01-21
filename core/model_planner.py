def plan_models(problem_info, llm_reasoning=None):
    """
    Rule-based model planning.
    Optionally enriches explanations using LLM reasoning (if provided).
    """
    plans = []

    def enrich(text):
        if llm_reasoning:
            return f"{text}\n\nLLM Insight: {llm_reasoning}"
        return text

    if problem_info["task_type"] == "classification":
        plans.append({
            "title": "Baseline Model: Logistic Regression",
            "reason": enrich(
                "Logistic Regression is simple, fast, and highly interpretable. "
                "It is ideal as a first model to understand feature importance "
                "and establish a baseline."
            ),
            "model": """
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
"""
        })

        plans.append({
            "title": "Advanced Model: Random Forest",
            "reason": enrich(
                "Random Forest captures non-linear relationships, "
                "handles feature interactions well, and is robust to outliers. "
                "Often performs better on tabular data."
            ),
            "model": """
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)
"""
        })

    elif problem_info["task_type"] == "regression":
        plans.append({
            "title": "Baseline Model: Linear Regression",
            "reason": enrich(
                "Linear Regression provides a strong interpretable baseline "
                "and helps understand linear relationships."
            ),
            "model": """
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
"""
        })

        plans.append({
            "title": "Advanced Model: XGBoost",
            "reason": enrich(
                "XGBoost handles complex non-linear patterns, "
                "feature interactions, and often achieves strong performance "
                "on structured data."
            ),
            "model": """
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)
"""
        })

    return plans
