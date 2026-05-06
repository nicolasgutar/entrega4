import pandas as pd
import numpy as np


def fill_missing_values(df: pd.DataFrame, method: str = "linear", limit: int = 6) -> pd.DataFrame:
    """
    Interpolate missing values. Linear interpolation for gaps <= limit hours;
    forward-fill fallback for remaining NaNs at edges.
    """
    df_filled = df.interpolate(method=method, limit=limit)
    df_filled = df_filled.ffill().bfill()
    nulls_before = df.isnull().sum().sum()
    nulls_after = df_filled.isnull().sum().sum()
    print(f"Valores nulos antes: {nulls_before:,}  →  después: {nulls_after:,}")
    return df_filled


def create_train_test_split(
    series: pd.Series,
    test_size: float = 0.15,
    verbose: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Temporal split — no shuffling."""
    n = len(series)
    split_idx = int(n * (1 - test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    if verbose:
        print(f"Train: {train.index.min()} → {train.index.max()}  ({len(train):,} obs)")
        print(f"Test : {test.index.min()} → {test.index.max()}  ({len(test):,} obs)")
        print(f"Split: {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")
    return train, test


def get_feature_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
