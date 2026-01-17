def get_problem_explanation(problem_info):
    """
    Structured explanation templates for problem understanding.
    """

    explanation = []

    explanation.append({
        "title": "Problem Type",
        "content": f"This is identified as a {problem_info['task_type']} problem."
    })

    explanation.append({
        "title": "Why This Decision",
        "content": problem_info.get("reasoning", "")
    })

    explanation.append({
        "title": "ML vs DL",
        "content": f"This problem is best solved using {problem_info['ml_type'].upper()} approaches."
    })

    return explanation
