import json
import pandas as pd
from llm_engine.llm_client import call_llm
from llm_engine.prompts import CLEANING_GUIDE_PROMPT


def _build_profile_digest(df, max_cols=12):
    return [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_pct": round(df[col].nunique() / max(len(df), 1) * 100, 2),
        }
        for col in df.columns[:max_cols]
    ]


def _generate_rule_based_cleaning(df):
    """
    Generates tailored, dataset-specific code when LLM is unavailable or unparsed.
    """
    # 1. Missing Values
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        missing_lines = ["# Handle missing values per column type"]
        for col in null_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                missing_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
            else:
                missing_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])")
        missing_code = "\n".join(missing_lines)
        missing_reason = f"Detected missing values in columns: {', '.join(null_cols)}."
    else:
        missing_code = "# Zero missing values detected across all columns!\n# No imputation required."
        missing_reason = "Dataset is complete with 0 missing values."

    # 2. Categorical Encoding
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        encoding_code = (
            f"# Convert categorical variables to binary indicators\n"
            f"categorical_cols = {cat_cols}\n"
            f"df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)"
        )
        encoding_reason = f"Identified {len(cat_cols)} non-numeric features requiring encoding for scikit-learn models."
    else:
        encoding_code = "# All features are already numeric.\n# No categorical encoding necessary."
        encoding_reason = "No categorical or string features detected."

    # 3. Feature Scaling
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if num_cols:
        scaling_code = (
            "from sklearn.preprocessing import StandardScaler\n\n"
            f"numeric_features = {num_cols[:8]}\n"
            "scaler = StandardScaler()\n"
            "df[numeric_features] = scaler.fit_transform(df[numeric_features])"
        )
        scaling_reason = "Scale continuous variables to have 0 mean and unit variance for distance and gradient-based models."
    else:
        scaling_code = "# No numeric features available for scaling."
        scaling_reason = "No numerical columns detected."

    return {
        "MISSING": {"code": missing_code, "reason": missing_reason},
        "ENCODING": {"code": encoding_code, "reason": encoding_reason},
        "SCALING": {"code": scaling_code, "reason": scaling_reason},
    }


def _split_sections(text):
    sections = {"MISSING": "", "ENCODING": "", "SCALING": ""}
    current = None
    buffer = []

    for line in text.splitlines():
        line_strip = line.strip()

        if line_strip == "[MISSING]":
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = "MISSING"
            buffer = []
            continue

        if line_strip == "[ENCODING]":
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = "ENCODING"
            buffer = []
            continue

        if line_strip == "[SCALING]":
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = "SCALING"
            buffer = []
            continue

        if current:
            buffer.append(line)

    if current:
        sections[current] = "\n".join(buffer).strip()

    return sections


def get_cleaning_guidance(df, session_state=None):
    if df is None:
        return []

    rule_fallback = _generate_rule_based_cleaning(df)

    digest = _build_profile_digest(df)
    llm_text = call_llm(
        prompt=CLEANING_GUIDE_PROMPT + "\n\nCOLUMN_METADATA:\n" + json.dumps(digest, indent=2),
        fallback_context="",
        cache_key="cleaning_no_outliers",
        session_state=session_state,
    )

    parts = _split_sections(llm_text) if llm_text else {}

    # Use LLM section if present and valid; otherwise use concrete dataset-aware fallback
    missing_code = parts.get("MISSING") or rule_fallback["MISSING"]["code"]
    missing_reason = "Dataset-aware guidance." if parts.get("MISSING") else rule_fallback["MISSING"]["reason"]

    encoding_code = parts.get("ENCODING") or rule_fallback["ENCODING"]["code"]
    encoding_reason = "Dataset-aware guidance." if parts.get("ENCODING") else rule_fallback["ENCODING"]["reason"]

    scaling_code = parts.get("SCALING") or rule_fallback["SCALING"]["code"]
    scaling_reason = "Dataset-aware guidance." if parts.get("SCALING") else rule_fallback["SCALING"]["reason"]

    return [
        {
            "title": "Handling Missing Values",
            "reason": missing_reason,
            "code": missing_code,
        },
        {
            "title": "Encoding Categorical Features",
            "reason": encoding_reason,
            "code": encoding_code,
        },
        {
            "title": "Feature Scaling",
            "reason": scaling_reason,
            "code": scaling_code,
        },
    ]
