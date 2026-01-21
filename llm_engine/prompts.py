def problem_understanding_prompt(user_goal, dataset_info):
    """
    Prompt for understanding ML problem type with low token usage.
    """

    prompt = f"""
You are a senior machine learning engineer.
Be concise. Do not add extra text.

User goal:
{user_goal}

Dataset metadata:
- Rows: {dataset_info['rows']}
- Columns: {dataset_info['columns']}
- Column names: {dataset_info['column_names']}

Tasks:
1. Decide ML or DL
2. Decide task type: classification / regression / time_series / other
3. Infer target variable type
4. Brief reasoning (1–2 sentences)

Rules:
- If data looks tabular → prefer ML
- If temporal patterns are likely → consider time_series
- Do NOT assume deep learning unless necessary

Return STRICT JSON only:
{{
  "ml_type": "",
  "task_type": "",
  "target_type": "",
  "reasoning": ""
}}
"""
    return prompt


def cleaning_suggestion_prompt(dataset_profile):
    """
    Prompt for LLM-assisted cleaning suggestions.
    Uses ONLY metadata (cheap, safe).
    """

    prompt = f"""
You are a senior data scientist.
Given dataset metadata ONLY (no raw values), suggest high-level data cleaning advice.

Dataset profile:
{dataset_profile}

Instructions:
- Do NOT suggest executing code
- Focus on reasoning (missing values, outliers, encoding, scaling)
- Mention risks or assumptions
- Keep it concise (4–5 bullet points)

Return plain text (NOT JSON).
"""
    return prompt
