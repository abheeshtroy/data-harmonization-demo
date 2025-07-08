# src/ai_agents/agents/prep_agent.py

from ai_agents.tools.etl import load_and_clean_all
from ai_agents.tools.prep_tool import prepare

def main():
    crm, ecom = load_and_clean_all()
    for combo in ["name", "address", "name_address"]:
        series_crm, series_ecom = prepare(crm, ecom, combo)
        print(f"\n=== {combo.upper()} ===")
        print("CRM example:", series_crm.iloc[0])
        print("E-com example:", series_ecom.iloc[0])

if __name__ == "__main__":
    main()
