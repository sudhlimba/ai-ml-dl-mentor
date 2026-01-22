from llm_engine.llm_client import call_llm


def _llm_train_test_reasoning(task_type, is_time_series):
    """
    LLM-assisted advisory reasoning only.
    Safe, low-token, optional.
    """
    prompt = f"""
You are a senior ML engineer.

Context:
- task_type: {task_type}
- is_time_series: {is_time_series}

Task:
Explain the most suitable train-test split strategy
and highlight any data leakage risks.

Rules:
- Advisory only (no execution)
- Do NOT override deterministic logic
- Max 4 bullet points
- Plain text only
"""
    return call_llm(prompt=prompt)


def get_train_test_guidance(task_type, is_time_series=False):
    """
    Returns structured guidance for choosing train-test split strategy.
    Teaching + planning only (no training).
    """

    guidance = []

    # ===============================
    # TIME SERIES SPLIT (PRIORITY)
    # ===============================
    if is_time_series:
        llm_reason = _llm_train_test_reasoning(task_type, is_time_series)

        guidance.append({
            "title": "Time Series Train-Test Split",
            "when": "Use when data points are ordered in time (sales, stock, sensors).",
            "why": (
                llm_reason
                if llm_reason
                else "Time-series data violates the i.i.d. assumption. "
                     "Random or stratified splits would leak future information."
            ),
            "code": """
# Sort data by time before splitting
df = df.sort_values(by="date_column")

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df  = df.iloc[split_index:]

# NEVER shuffle time-series data
""".strip(),
            "source": "llm" if llm_reason else "rule"
        })

        return guidance

    # ===============================
    # RANDOM SPLIT (NOW LLM-AWARE)
    # ===============================
    llm_reason = _llm_train_test_reasoning(task_type, is_time_series)

    guidance.append({
        "title": "Random Train-Test Split",
        "when": "Use when data rows are independent and order does not matter.",
        "why": (
            llm_reason
            if llm_reason
            else "Ensures train and test sets come from the same distribution."
        ),
        "code": """
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
""".strip(),
        "source": "llm" if llm_reason else "rule"
    })

    # ===============================
    # STRATIFIED SPLIT (CLASSIFICATION)
    # ===============================
    if task_type == "classification":
        llm_reason = _llm_train_test_reasoning(task_type, is_time_series)

        guidance.append({
            "title": "Stratified Train-Test Split",
            "when": "Use when classification labels are imbalanced.",
            "why": (
                llm_reason
                if llm_reason
                else "Preserves class distribution in both train and test sets."
            ),
            "code": """
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
""".strip(),
            "source": "llm" if llm_reason else "rule"
        })

    return guidance
