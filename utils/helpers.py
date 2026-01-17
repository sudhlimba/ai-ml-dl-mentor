def is_csv_file(filename):
    if not filename:
        return False
    return filename.lower().endswith(".csv")


def safe_lower(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()
