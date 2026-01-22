import os
import openai
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# LLM SAFETY & COST CONTROLS
# -------------------------------
MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 1200          # hard cap to control cost
TEMPERATURE = 0.2
REQUEST_TIMEOUT = 10     # seconds


def call_llm(prompt, fallback_context=None, cache_key=None, session_state=None):
    
    """
    Tries OpenAI first (if API key exists).
    Enforces token limits & safe defaults.
    If API fails → returns rule-based safe response.
    """

    # -------------------------------
    # OPTIONAL CACHE (PER SESSION)
    # -------------------------------
    if cache_key and session_state is not None:
        if cache_key in session_state:
            return session_state[cache_key]

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        
        try:
            print("✅ OPENAI API USED")
            openai.api_key = api_key

            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a careful ML mentor. Be concise and practical."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )

            content = response["choices"][0]["message"]["content"]

            # store in cache if provided
            if cache_key and session_state is not None:
                session_state[cache_key] = content

            return content

        except Exception:
            print("❌ OPENAI FAILED, FALLING BACK")
            pass  # silently fall back (never crash app)

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
  "reasoning": "The goal describes a classification problem with categorical outcomes. Interpretable ML models are appropriate."
}
"""

    if "regression" in goal or "forecast" in goal:
        return """
{
  "ml_type": "ml",
  "task_type": "regression",
  "target_type": "numerical",
  "reasoning": "The goal involves predicting a continuous numeric value, which aligns with regression modeling."
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
