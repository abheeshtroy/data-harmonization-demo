# src/ai_agents/tools/prep_tool.py

from ai_agents.tools.etl import prep_name, prep_address, prep_name_address
import pandas as pd

def prepare(crm: pd.DataFrame, ecom: pd.DataFrame, combo: str):
    """
    Given the cleaned CRM and E-com DataFrames and a combo key,
    return two pandas.Series of strings ready for embedding.
    combo must be one of: "name", "address", "name_address".
    """
    if combo == "name":
        return prep_name(crm), prep_name(ecom)
    elif combo == "address":
        return prep_address(crm), prep_address(ecom)
    elif combo == "name_address":
        return prep_name_address(crm), prep_name_address(ecom)
    else:
        raise ValueError(f"Unknown combo '{combo}'")
