# src/etl.py
import pandas as pd

# Define valid boroughs
VALID_BOROUGHS = {"Manhattan","Brooklyn","Queens","Bronx","Staten Island"}

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop duplicates
    df = df.drop_duplicates(subset="id")

    # 2. Fill text fields
    df["name"] = df["name"].fillna("").astype(str)
    df["host_name"] = df["host_name"].fillna("unknown").astype(str)

    # 3. Borough standardization
    df["neighbourhood_group"] = df["neighbourhood_group"]\
        .where(df["neighbourhood_group"].isin(VALID_BOROUGHS), other="Other")

    # 4. Neighborhood—group rare ones
    counts = df["neighbourhood"].value_counts()
    rare = counts[counts < 50].index
    df["neighbourhood"] = df["neighbourhood"]\
        .where(~df["neighbourhood"].isin(rare), other="Other")

    # 5. Price outliers
    #   a) Clip any price above $1000 to 1000
    df["price"] = df["price"].clip(upper=1000)
    #   b) (Optional) or: df["price_log"] = np.log1p(df["price"])

    # 6. Reviews/date
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)

    # 7. Geolocation sanity check (optional)
    df = df[
        (df["latitude"].between(40.5, 40.92)) &
        (df["longitude"].between(-74.25, -73.7))
    ]

    return df
