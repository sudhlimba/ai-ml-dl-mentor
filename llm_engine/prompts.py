def problem_understanding_prompt(user_goal, dataset_info):
    """
    Prompt for understanding ML/DL problem type.
    """

    prompt = f"""
You are a senior machine learning engineer.

User Goal:
{user_goal}

Dataset Info:
Rows: {dataset_info['rows']}
Columns: {dataset_info['columns']}
Column Names: {dataset_info['column_names']}

Decide:
1. ML or DL
2. Task type (classification / regression / other)
3. Target variable type
4. Reasoning

Return STRICT JSON like this:
{{
  "ml_type": "",
  "task_type": "",
  "target_type": "",
  "reasoning": ""
}}
"""
    return prompt
