import pandas as pd
import numpy as np


def profile_dataset(df):
    """
    Create a health report of the dataset.
    """
    report = []

    for col in df.columns:
        col_data = df[col]

        col_info = {
            "column": col,
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "non_null_count": int(col_data.notnull().sum()),
            "unique_values": int(col_data.nunique())
        }

        if pd.api.types.is_numeric_dtype(col_data):
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = col_data[(col_data < lower) | (col_data > upper)]
            col_info["outlier_count"] = int(outliers.count())
        else:
            col_info["outlier_count"] = "N/A"

        report.append(col_info)

    return pd.DataFrame(report)
