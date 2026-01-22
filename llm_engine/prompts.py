# ===============================
# PROBLEM UNDERSTANDING PROMPT
# ===============================
def problem_understanding_prompt(user_goal, dataset_info):
    prompt = f"""
You are a senior machine learning engineer.

User goal:
{user_goal}

Dataset:
- Rows: {dataset_info['rows']}
- Columns: {dataset_info['columns']}
- Column names: {dataset_info['column_names']}

Return STRICT JSON:
{{
  "ml_type": "",
  "task_type": "",
  "target_type": "",
  "reasoning": ""
}}
"""
    return prompt


# ===============================
# CLEANING — NO OUTLIERS
# ===============================
CLEANING_GUIDE_PROMPT = """
You are a senior data scientist mentoring a junior ML engineer.

You will receive dataset column metadata.

Return guidance STRICTLY in this format.
Do NOT include any Outliers section.

[MISSING]
Columns:
- columns with missing values OR None

Method:
- per-column strategy (median / mode / drop / none)

Why:
- short explanation

Example:
- REQUIRED if Columns ≠ None
- else write: Not required

[ENCODING]
Columns:
- categorical columns only

Method:
- specify per column (one-hot / binary)

Why:
- short explanation

Example:
- REQUIRED pandas code
- must be COMPLETE (no truncation)

[SCALING]
Columns:
- numeric columns that benefit from scaling
- exclude binary / target-like columns

Method:
- standard / robust / minmax (per column)

Why:
- short explanation

Example:
- REQUIRED sklearn code
- must be COMPLETE

Rules:
- NEVER invent columns
- NEVER output partial code
- Assume dataframe name is df
- Plain text only
"""
