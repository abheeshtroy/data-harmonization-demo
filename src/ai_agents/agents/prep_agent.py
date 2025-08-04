# src/ai_agents/agents/prep_agent.py
import pandas as pd
from typing import Union, List, Tuple
from ai_agents.tools.etl import prep_name, prep_email, prep_phone, prep_address

def prep_texts(
    crm_df: pd.DataFrame,
    ecom_df: pd.DataFrame,
    combo: Union[str, List[str]]
) -> Tuple[pd.Series, pd.Series]:
    """
    Given cleaned CRM and E-com DataFrames and a combo key or list of keys,
    return two pandas.Series ready for embedding.
    combo can be:
      - "all"               → ["name","email","phone","address"]
      - a single field name
      - a list of field names
    """
    # Normalize to a list
    if isinstance(combo, str):
        combos = [combo]
    else:
        combos = combo[:]

    # Expand "all"
    if "all" in combos:
        combos = ["name", "email", "phone", "address"]

    # Validate
    allowed = {"name", "email", "phone", "address"}
    for c in combos:
        if c not in allowed:
            raise ValueError(f"Unknown combo '{c}'; choose from {allowed} or 'all'")

    # Build lists of Series
    seq_crm: List[pd.Series] = []
    seq_ecom: List[pd.Series] = []
    for field in combos:
        if field == "name":
            seq_crm.append(prep_name(crm_df))
            seq_ecom.append(prep_name(ecom_df))
        elif field == "email":
            seq_crm.append(prep_email(crm_df))
            seq_ecom.append(prep_email(ecom_df))
        elif field == "phone":
            seq_crm.append(prep_phone(crm_df))
            seq_ecom.append(prep_phone(ecom_df))
        elif field == "address":
            seq_crm.append(prep_address(crm_df))
            seq_ecom.append(prep_address(ecom_df))

    # Zip together each row’s parts with a newline, for embedding
    def zip_and_join(series_list):
        # convert each Series to list, then zip
        lists = [s.tolist() for s in series_list]
        joined = ["\n".join(parts) for parts in zip(*lists)]
        return pd.Series(joined)

    return zip_and_join(seq_crm), zip_and_join(seq_ecom)
