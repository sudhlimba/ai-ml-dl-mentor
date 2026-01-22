from llm_engine.llm_client import call_llm
from llm_engine.prompts import CLEANING_GUIDE_PROMPT
import json


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

    digest = _build_profile_digest(df)

    llm_text = call_llm(
        prompt=CLEANING_GUIDE_PROMPT + "\n\nCOLUMN_METADATA:\n" + json.dumps(digest, indent=2),
        fallback_context="",
        cache_key="cleaning_no_outliers",
        session_state=session_state,
    )

    if not llm_text:
        return [
            {"title": "Handling Missing Values", "reason": "Rule-based.", "code": "Median / Mode"},
            {"title": "Encoding Categorical Features", "reason": "Rule-based.", "code": "One-hot encoding"},
            {"title": "Feature Scaling", "reason": "Rule-based.", "code": "StandardScaler"},
        ]

    parts = _split_sections(llm_text)

    return [
        {
            "title": "Handling Missing Values",
            "reason": "Dataset-aware guidance.",
            "code": parts["MISSING"],
        },
        {
            "title": "Encoding Categorical Features",
            "reason": "Dataset-aware guidance.",
            "code": parts["ENCODING"],
        },
        {
            "title": "Feature Scaling",
            "reason": "Dataset-aware guidance.",
            "code": parts["SCALING"],
        },
    ]
