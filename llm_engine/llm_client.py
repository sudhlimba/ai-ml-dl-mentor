import os
import openai
from dotenv import load_dotenv

load_dotenv()


def call_llm(prompt, fallback_context=None):
    """
    Tries OpenAI first.
    If quota / API fails → returns rule-based safe response.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            openai.api_key = api_key

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a careful ML mentor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            return response["choices"][0]["message"]["content"]

        except Exception:
            pass  # silently fall back

    # ===============================
    # FALLBACK (NO API REQUIRED)
    # ===============================
    goal = (fallback_context or "").lower()

    if "predict" in goal or "classification" in goal or "heart" in goal:
        return """
{
  "ml_type": "ml",
  "task_type": "classification",
  "target_type": "categorical",
  "reasoning": "The goal describes a binary outcome prediction problem. Traditional ML classification models are sufficient and interpretable for medical risk prediction."
}
"""

    if "regression" in goal or "forecast" in goal:
        return """
{
  "ml_type": "ml",
  "task_type": "regression",
  "target_type": "numerical",
  "reasoning": "The goal involves predicting a continuous numeric value, which is best handled using regression models."
}
"""

    return """
{
  "ml_type": "ml",
  "task_type": "classification",
  "target_type": "unknown",
  "reasoning": "Fallback decision based on common supervised learning patterns."
}
"""
