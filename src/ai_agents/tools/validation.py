import pandas as pd

def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one‐row summary: #rows, #cols, memory usage."""
    return pd.DataFrame({
        "rows": [df.shape[0]],
        "cols": [df.shape[1]],
        "memory_MB": [df.memory_usage(deep=True).sum() / 1e6]
    })

def null_counts(df: pd.DataFrame) -> pd.Series:
    """Return count of nulls per column."""
    return df.isnull().sum()

def duplicate_count(df: pd.DataFrame, subset=None) -> int:
    """Return number of duplicate rows (optionally on subset of columns)."""
    return df.duplicated(subset=subset).sum()
