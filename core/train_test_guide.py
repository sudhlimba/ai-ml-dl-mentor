def get_train_test_guidance(task_type, has_time_column=False):
    """
    Returns structured guidance for choosing train-test split strategy.
    Teaching + planning only (no training).
    """

    guidance = []

    # ===============================
    # RANDOM SPLIT
    # ===============================
    guidance.append({
        "title": "Random Train-Test Split",
        "when": "Use when data rows are independent and order does not matter.",
        "why": "Ensures train and test sets come from the same distribution.",
        "code": """
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Suitable for most tabular ML problems
""".strip()
    })

    # ===============================
    # STRATIFIED SPLIT
    # ===============================
    if task_type == "classification":
        guidance.append({
            "title": "Stratified Train-Test Split",
            "when": "Use when classification labels are imbalanced.",
            "why": "Preserves class distribution in both train and test sets.",
            "code": """
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Recommended for medical and risk prediction datasets
""".strip()
        })

    # ===============================
    # TIME SERIES SPLIT
    # ===============================
    if has_time_column:
        guidance.append({
            "title": "Time Series Split",
            "when": "Use when data has a time order (sales, stock, sensor data).",
            "why": "Prevents future data from leaking into training.",
            "code": """
# Sort data by time before splitting
df = df.sort_values(by="date_column")

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df  = df.iloc[split_index:]

# Never shuffle time-series data
""".strip()
        })

    return guidance
