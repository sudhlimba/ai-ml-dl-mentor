from llm_engine.llm_client import call_llm


def _llm_model_planning_reasoning(problem_info):
    """
    LLM-assisted advisory reasoning for model planning.
    """
    prompt = f"""
You are a senior ML engineer.

Context:
- task_type: {problem_info.get("task_type")}
- dataset_type: tabular
- goal: interview-safe, explainable ML

Explain:
1. Why a simple baseline model is suitable
2. Why a stronger final model is suitable

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

    if llm_reasoning is None:
        llm_reasoning = _llm_model_planning_reasoning(problem_info)

    task_type = problem_info.get("task_type")

    # ===============================
    # CLASSIFICATION
    # ===============================
    if task_type == "classification":
        plans.append({
            "title": "Baseline Model",
            "reason": (
                "Model: Logistic Regression\n\n"
                "Why:\n"
                "- Simple and fast baseline\n"
                "- Highly interpretable\n"
                "- Good reference for comparison\n\n"
                f"LLM Insight:\n{llm_reasoning}"
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
                "Model: Random Forest Classifier\n\n"
                "Why:\n"
                "- Captures non-linear relationships\n"
                "- Handles feature interactions\n"
                "- Strong performance on tabular data\n\n"
                f"LLM Insight:\n{llm_reasoning}"
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
        plans.append({
            "title": "Baseline Model",
            "reason": (
                "Model: Linear Regression\n\n"
                "Why:\n"
                "- Simple, interpretable baseline\n"
                "- Helps understand linear relationships\n\n"
                f"LLM Insight:\n{llm_reasoning}"
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
                "Model: XGBoost Regressor\n\n"
                "Why:\n"
                "- Captures complex non-linear patterns\n"
                "- Handles feature interactions well\n"
                "- Strong performance on structured data\n\n"
                f"LLM Insight:\n{llm_reasoning}"
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
        plans.append({
            "title": "Baseline Model",
            "reason": (
                "Model: Logistic / Linear Regression\n\n"
                "Why:\n"
                "- Task type not clearly inferred\n"
                "- Simple baseline to understand data behavior\n\n"
                f"LLM Insight:\n{llm_reasoning}"
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
                "Model: Tree-based Model\n\n"
                "Why:\n"
                "- More expressive than linear models\n"
                "- Good next step after baseline\n\n"
                f"LLM Insight:\n{llm_reasoning}"
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
