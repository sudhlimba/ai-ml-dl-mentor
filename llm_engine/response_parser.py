import json


def parse_llm_response(raw_text):
    """
    Safely parse JSON response from LLM (problem understanding).
    """
    try:
        data = json.loads(raw_text)
        return {
            "ml_type": data.get("ml_type", "").lower(),
            "task_type": data.get("task_type", "").lower(),
            "target_type": data.get("target_type", "").lower(),
            "reasoning": data.get("reasoning", "")
        }
    except Exception:
        return {
            "ml_type": "ml",
            "task_type": "classification",
            "target_type": "unknown",
            "reasoning": "Failed to parse LLM response. Using safe defaults."
        }


# ===============================
# NEW — Cleaning response parser
# ===============================
def parse_cleaning_response(raw_text):
    """
    Parse LLM cleaning suggestions.
    Expected STRICT JSON.
    Fallback-safe.
    """
    try:
        data = json.loads(raw_text)

        if not isinstance(data, dict):
            return None

        columns = data.get("columns", {})
        if not isinstance(columns, dict):
            return None

        steps = []
        for col, suggestions in columns.items():
            if suggestions == "none" or not suggestions:
                continue

            steps.append({
                "title": f"Cleaning suggestions for '{col}'",
                "reason": "LLM-assisted dataset-aware recommendation.",
                "code": "\n".join([f"- {s}" for s in suggestions])
            })

        return steps if steps else None

    except Exception:
        return None
