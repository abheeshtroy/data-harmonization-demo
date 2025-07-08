import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """Simple CSV loader abstraction."""
    return pd.read_csv(path)
