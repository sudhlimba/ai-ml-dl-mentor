import pandas as pd


def load_csv(file):
    """
    Safely load a CSV file into a pandas DataFrame.
    No assumptions, no preprocessing.
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError("Unable to read the CSV file")


def get_basic_info(df):
    """
    Return basic dataset information.
    """
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return info
