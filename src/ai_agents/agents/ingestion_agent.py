# src/ai_agents/agents/ingestion_agent.py

from ai_agents.tools.etl import load_and_clean_all
import pandas as pd

def load_crm() -> pd.DataFrame:
    """
    Load and clean the CRM dataset, returning a DataFrame
    with columns renamed to our unified schema.
    """
    crm, _ = load_and_clean_all()
    return crm

def load_ecom() -> pd.DataFrame:
    """
    Load and clean the E‐commerce dataset, returning a DataFrame
    with columns renamed to our unified schema.
    """
    _, ecom = load_and_clean_all()
    return ecom
