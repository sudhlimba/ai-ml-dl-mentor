def plan_models(problem_info):
    plans = []

    if problem_info["task_type"] == "classification":
        plans.append({
            "title": "Baseline Model: Logistic Regression",
            "reason": (
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
            "reason": (
                "Random Forest captures non-linear relationships, "
                "handles feature interactions well, and is robust to outliers. "
                "Often performs significantly better on tabular medical data."
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
            "reason": (
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
            "reason": (
                "XGBoost handles complex non-linear patterns, "
                "feature interactions, and typically achieves state-of-the-art "
                "performance on structured data."
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
