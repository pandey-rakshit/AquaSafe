import pandas as pd

def validate_input_df(df: pd.DataFrame, required_features: list):
    missing = set(required_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    if df.isna().any().any():
        raise ValueError("Input contains missing values")
