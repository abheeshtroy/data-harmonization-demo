# src/ai_agents/experiments/evaluate_match.py

import pandas as pd
from ai_agents.tools.etl import get_ground_truth

def main():
    # 1) load our ground truth (CRM→E-com correct pairs)
    gt = get_ground_truth(n=None)  # all pairs
    #    ground truth has columns: customer_id, name_crm, name_ecom, address_crm, address_ecom

    # 2) load our predicted top-1 matches
    preds = pd.read_csv("data/customer_matches.csv")

    # 3) merge on CRM ID
    df = preds.merge(
        gt[["customer_id", "name_ecom"]].rename(columns={
            "customer_id": "crm_id",
            "name_ecom":  "true_ecom_name"
        }),
        on="crm_id",
        how="left"
    )

    # 4) also attach true E-com ID by looking up where name matches
    #    (ground truth DataFrame uses e-com name but no ID column, so we rebuild a small map)
    gt_id_map = gt.assign(ecom_id=gt["customer_id"]) \
                  .set_index("name_ecom")["ecom_id"].to_dict()

    df["true_ecom_id"] = df["true_ecom_name"].map(gt_id_map)

    # 5) compute accuracy = fraction where our ecom_id == true_ecom_id
    df["correct"] = df["ecom_id"] == df["true_ecom_id"]
    accuracy = df["correct"].mean()

    # 6) report
    print(f"Total evaluated: {len(df)}")
    print(f"Correct top-1 matches: {df['correct'].sum()}  ({accuracy:.3%})\n")
    print("Examples wrong matches:")
    print(df[~df["correct"]].head()[[
        "crm_id", "ecom_id", "true_ecom_id", "crm_idx", "ecom_idx"
    ]])

if __name__ == "__main__":
    main()
