from ai_agents.tools.etl import load_and_clean_all
from ai_agents.tools.validation import summarize_df, null_counts, duplicate_count

def run_validation():
    crm, ecom = load_and_clean_all()
    print("=== CRM Summary ===")
    print(summarize_df(crm))
    print("\nNulls in CRM:")
    print(null_counts(crm))
    print("Duplicates in CRM:", duplicate_count(crm, subset=["customer_id"]))

    print("\n=== E-com Summary ===")
    print(summarize_df(ecom))
    print("\nNulls in E-com:")
    print(null_counts(ecom))
    print("Duplicates in E-com:", duplicate_count(ecom, subset=["customer_id"]))

if __name__ == "__main__":
    run_validation()