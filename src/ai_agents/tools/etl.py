import pandas as pd
import re
from pathlib import Path

# ─── locate data files ─────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[3]
DATA_DIR  = BASE_DIR / "data"
CRM_PATH  = DATA_DIR / "dataset1_crm.csv"
ECOM_PATH = DATA_DIR / "dataset2_ecommerce.csv"

# ─── phone normalization ─────────────────────────────────────────────────────────
_phone_re = re.compile(r"\D+")

def _normalize_phone(s: str) -> str:
    """Strip non-digits, drop leading '1' if present, return exactly 10 digits."""
    digits = _phone_re.sub("", s or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits.zfill(10)

# ─── cleaning & loading for CRM ───────────────────────────────────────────────────
def load_crm() -> pd.DataFrame:
    return pd.read_csv(CRM_PATH)

def clean_crm(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Rename to unified schema
    df = df.rename(columns={
        "source_id":         "id",
        "full_name":         "name",
        "email_address":     "email",
        "phone":             "phone",
        "street_address":    "address",
        "company":           "org",
        "job_title":         "role",
        "birth_date":        "dob",
        "registration_date": "signup",
        "status":            "status",
        "score":             "score",
        "category":          "group",
    })

    # 2) Lower-case & trim
    for col in ["name", "email", "address", "org", "role", "status", "group"]:
        df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

    # 3) Phone → 10 digits
    df["phone"] = df["phone"].apply(_normalize_phone)

    # 4) Parse dates
    df["dob"]    = pd.to_datetime(df["dob"],    errors="coerce")
    df["signup"] = pd.to_datetime(df["signup"], errors="coerce")

    # 5) Score → numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

    return df

# ─── cleaning & loading for E-commerce ──────────────────────────────────────────
def load_ecom() -> pd.DataFrame:
    return pd.read_csv(ECOM_PATH)

def clean_ecom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "target_id":       "id",
        "customer_name":   "name",
        "contact_email":   "email",
        "telephone":       "phone",
        "mailing_address": "address",
        "organization":    "org",
        "position":        "role",
        "date_of_birth":   "dob",
        "signup_date":     "signup",
        "account_status":  "status",
        "rating":          "score",
        "tier":            "group",
        "region":          "region",
    })

    for col in ["name", "email", "address", "org", "role", "status", "group", "region"]:
        df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

    df["phone"] = df["phone"].apply(_normalize_phone)
    df["dob"]    = pd.to_datetime(df["dob"],    errors="coerce")
    df["signup"] = pd.to_datetime(df["signup"], errors="coerce")
    df["score"]  = pd.to_numeric(df["score"], errors="coerce").fillna(0)

    return df

# ─── helper: load & clean both tables ─────────────────────────────────────────────
def load_and_clean_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    crm  = clean_crm(load_crm())
    ecom = clean_ecom(load_ecom())
    return crm, ecom

# ─── prep helpers ────────────────────────────────────────────────────────────────
def prep_name(df: pd.DataFrame) -> pd.Series:
    s = df["name"].fillna("").astype(str).str.strip().str.lower()
    def swap(x: str) -> str:
        if "," in x:
            last, first = [p.strip() for p in x.split(",", 1)]
            return f"{first} {last}"
        return x
    return s.apply(swap)

def prep_email(df: pd.DataFrame) -> pd.Series:
    return df["email"].fillna("").astype(str).str.strip().str.lower()

def prep_phone(df: pd.DataFrame) -> pd.Series:
    s = df["phone"].fillna("").astype(str)
    digits = s.str.replace(r"\D+", "", regex=True)
    digits = digits.apply(lambda x: x[1:] if len(x)==11 and x.startswith("1") else x)
    return digits

def prep_address(df: pd.DataFrame) -> pd.Series:
    s = df["address"].fillna("").astype(str).str.strip().str.lower()
    s = s.str.replace("street", "st",    regex=False)
    s = s.str.replace("avenue", "ave",   regex=False)
    s = s.str.replace("road", "rd",      regex=False)
    s = s.str.replace("drive", "dr",     regex=False)
    s = s.str.replace(r"[^\w\s]", " ",   regex=True)
    s = s.str.replace(r"\s+", " ",       regex=True)
    return s
