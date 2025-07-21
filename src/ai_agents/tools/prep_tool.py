import pandas as pd
from ai_agents.tools.etl import (
    load_and_clean_all,
    prep_name,
    prep_email,
    prep_phone,
    prep_address,
)

def prepare(crm: pd.DataFrame, ecom: pd.DataFrame, combo: str):
    """
    Return two pd.Series of strings to embed, based on combo:
      - "name", "email", "phone", "address", or "all"
    """
    # allow passing file paths instead of DataFrames
    if isinstance(crm, str) and isinstance(ecom, str):
        crm, ecom = load_and_clean_all()

    if combo == "name":
        return prep_name(crm), prep_name(ecom)

    if combo == "email":
        return prep_email(crm), prep_email(ecom)

    if combo == "phone":
        return prep_phone(crm), prep_phone(ecom)

    if combo == "address":
        return prep_address(crm), prep_address(ecom)

    if combo == "all":
        sep = " | "
        crm_text = (
            prep_name(crm)   + sep +
            prep_email(crm)  + sep +
            prep_phone(crm)  + sep +
            prep_address(crm)
        )
        ecom_text = (
            prep_name(ecom)   + sep +
            prep_email(ecom)  + sep +
            prep_phone(ecom)  + sep +
            prep_address(ecom)
        )
        return crm_text, ecom_text

    raise ValueError(
        f"Unknown combo '{combo}'. Choose from "
        "['name','email','phone','address','all']."
    )
