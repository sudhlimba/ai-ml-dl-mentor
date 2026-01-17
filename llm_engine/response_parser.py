import json


def parse_llm_response(raw_text):
    """
    Safely parse JSON response from LLM.
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
