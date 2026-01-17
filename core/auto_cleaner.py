import pandas as pd
from sklearn.preprocessing import StandardScaler
def auto_clean_dataframe(df):
    df = df.copy()
    # Handle missing values
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    # Feature scaling
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
