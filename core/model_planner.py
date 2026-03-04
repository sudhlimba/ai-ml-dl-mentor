from llm_engine.llm_client import call_llm


def _llm_model_reasoning(problem_info, model_name, model_role):
    """
    LLM-assisted advisory reasoning for a single model.
    model_role: "baseline" or "final"
    """
    prompt = f"""
You are a senior ML engineer.

Context:
- task_type: {problem_info.get("task_type")}
- dataset_type: tabular
- goal: interview-safe, explainable ML

Model: {model_name}
Role: {model_role} model

Explain why {model_name} is suitable as a {model_role} model for this task.

Rules:
- No AutoML
- No hyperparameter tuning
- Keep it concise
- Plain text only
"""
    return call_llm(prompt=prompt)


def plan_models(problem_info, llm_reasoning=None):
    """
    Always returns exactly TWO model plans:
    1) Baseline model
    2) Final model
    """

    plans = []
    task_type = problem_info.get("task_type")

    # ===============================
    # CLASSIFICATION
    # ===============================
    if task_type == "classification":

        baseline_model = "Logistic Regression"
        final_model = "Random Forest Classifier"

        baseline_reason = _llm_model_reasoning(
            problem_info,
            baseline_model,
            "baseline"
        )

        final_reason = _llm_model_reasoning(
            problem_info,
            final_model,
            "final"
        )

        plans.append({
            "title": "Baseline Model",
            "reason": (
                f"Model: {baseline_model}\n\n"
                f"{baseline_reason}"
            ),
            "model": """
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
"""
        })

        plans.append({
            "title": "Final Model",
            "reason": (
                f"Model: {final_model}\n\n"
                f"{final_reason}"
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

    # ===============================
    # REGRESSION
    # ===============================
    elif task_type == "regression":

        baseline_model = "Linear Regression"
        final_model = "XGBoost Regressor"

        baseline_reason = _llm_model_reasoning(
            problem_info,
            baseline_model,
            "baseline"
        )

        final_reason = _llm_model_reasoning(
            problem_info,
            final_model,
            "final"
        )

        plans.append({
            "title": "Baseline Model",
            "reason": (
                f"Model: {baseline_model}\n\n"
                f"{baseline_reason}"
            ),
            "model": """
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
"""
        })

        plans.append({
            "title": "Final Model",
            "reason": (
                f"Model: {final_model}\n\n"
                f"{final_reason}"
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

    # ===============================
    # FALLBACK (TASK UNCLEAR)
    # ===============================
    else:

        baseline_model = "Logistic / Linear Regression"
        final_model = "Tree-based Model"

        baseline_reason = _llm_model_reasoning(
            problem_info,
            baseline_model,
            "baseline"
        )

        final_reason = _llm_model_reasoning(
            problem_info,
            final_model,
            "final"
        )

        plans.append({
            "title": "Baseline Model",
            "reason": (
                f"Model: {baseline_model}\n\n"
                f"{baseline_reason}"
            ),
            "model": """
# Classification
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
"""
        })

        plans.append({
            "title": "Final Model",
            "reason": (
                f"Model: {final_model}\n\n"
                f"{final_reason}"
            ),
            "model": """
# Classification
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# Regression
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
"""
        })

    return plans