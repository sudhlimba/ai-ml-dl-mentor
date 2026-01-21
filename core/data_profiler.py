import pandas as pd
import numpy as np

from llm_engine.llm_client import call_llm


def profile_dataset(df):
    """
    Create a health report of the dataset.
    """
    report = []

    for col in df.columns:
        col_data = df[col]

        col_info = {
            "column": col,
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "non_null_count": int(col_data.notnull().sum()),
            "unique_values": int(col_data.nunique())
        }

        if pd.api.types.is_numeric_dtype(col_data):
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = col_data[(col_data < lower) | (col_data > upper)]
            col_info["outlier_count"] = int(outliers.count())
        else:
            col_info["outlier_count"] = "N/A"

        report.append(col_info)

    return pd.DataFrame(report)


# ==================================================
# TIME SERIES DETECTION (RULE-BASED + OPTIONAL LLM)
# ==================================================
def detect_time_series(df, session_state=None):
    """
    Detects whether dataset is time-series.
    Uses cheap rules first, then optional LLM reasoning (metadata only).
    """

    # ---------- Rule-based checks ----------
    datetime_cols = []
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notnull().mean() > 0.8:
                datetime_cols.append(col)
        except Exception:
            pass

    rule_based = bool(datetime_cols)

    result = {
        "is_time_series": rule_based,
        "datetime_columns": datetime_cols,
        "source": "rule"
    }

    # ---------- Optional LLM advisory ----------
    if session_state is not None and "time_series_check" not in session_state:
        meta = {
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "rows": df.shape[0]
        }

        prompt = f"""
You are a senior ML engineer.
Given only metadata (no data values), decide if this dataset likely represents time-series data.

Metadata:
{meta}

Answer strictly as JSON:
{{
  "is_time_series": true/false,
  "reasoning": "one sentence"
}}
"""

        raw = call_llm(prompt)
        session_state["time_series_check"] = raw

        # advisory only
        result["llm_advice"] = raw
        result["source"] = "rule+llm"

    return result
