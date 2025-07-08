# src/ai_agents/tools/etl.py

import pandas as pd
from pathlib import Path

# ─── Locate data directory ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

CRM_PATH  = DATA_DIR / "dataset1_crm.csv"
ECOM_PATH = DATA_DIR / "dataset2_ecommerce.csv"


# ─── 1) LOADERS ─────────────────────────────────────────────────────────────────
def load_crm() -> pd.DataFrame:
    """Load CRM CSV and return raw DataFrame."""
    return pd.read_csv(CRM_PATH)

  
def load_ecom() -> pd.DataFrame:
    """Load E-commerce CSV and return raw DataFrame."""
    return pd.read_csv(ECOM_PATH)


# ─── 2) CLEANERS / STANDARDIZERS ──────────────────────────────────────────────────
def clean_crm(df: pd.DataFrame) -> pd.DataFrame:
    # Rename into our canonical schema
    df = df.rename(columns={
        "customer_id":      "customer_id",
        "full_name":        "name",
        "email_address":    "email",
        "phone_number":     "phone",
        "address":          "address",
        "registration_date":"registration_date",
        "status":           "status",
        "total_purchases":  "order_value",
    })

    # Normalize text fields
    for col in ["name", "email", "phone", "address"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Parse dates and numbers
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["order_value"]       = pd.to_numeric(df["order_value"], errors="coerce").fillna(0)
    return df


def clean_ecom(df: pd.DataFrame) -> pd.DataFrame:
    # Rename into our canonical schema, including mapping order_date → registration_date
    df = df.rename(columns={
        "cust_id":        "customer_id",
        "customer_name":  "name",
        "contact_email":  "email",
        "phone_contact":  "phone",
        "order_date":     "registration_date",
        "account_type":   "status",
        "order_value":    "order_value",
    })
    # Build a unified `address`
    df["address"] = (
        df.get("shipping_street", "").fillna("") + ", "
      + df.get("shipping_city", "").fillna("")   + ", "
      + df.get("shipping_state", "").fillna("")  + " "
      + df.get("zip_code", "").astype(str).fillna("")
    ).str.strip(", ")

    # Normalize text fields
    for col in ["name", "email", "phone", "address"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Parse dates and numbers
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["order_value"]       = pd.to_numeric(df["order_value"], errors="coerce").fillna(0)
    return df


# ─── 3) COMBINED HELPERS ─────────────────────────────────────────────────────────
def load_and_clean_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both sources, clean them, and return (crm_df, ecom_df)."""
    crm  = clean_crm(load_crm())
    ecom = clean_ecom(load_ecom())
    return crm, ecom


def get_ground_truth(n: int = 20) -> pd.DataFrame:
    """
    Inner-join on customer_id to produce a ground-truth DataFrame
    with suffixes _crm and _ecom, then return the first n rows.
    """
    crm, ecom = load_and_clean_all()
    merged = crm.merge(
        ecom,
        on="customer_id",
        how="inner",
        suffixes=("_crm", "_ecom"),
    )
    return merged.head(n)


# ─── 4) PREP FUNCTIONS ───────────────────────────────────────────────────────────
def prep_name(df: pd.DataFrame) -> pd.Series:
    """Lowercase & strip the `name` column."""
    return df["name"].str.lower().str.strip()


def prep_address(df: pd.DataFrame) -> pd.Series:
    """Lowercase & strip the `address` column."""
    return df["address"].str.lower().str.strip()


def prep_name_address(df: pd.DataFrame) -> pd.Series:
    """Concatenate name + address (with “; ”) for joint embedding."""
    return prep_name(df) + "; " + prep_address(df)
