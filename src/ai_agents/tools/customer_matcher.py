# src/ai_agents/tools/customer_matcher.py

import pandas as pd
from ai_agents.tools.etl import load_and_clean_all, prep_address
from ai_agents.tools.embeddings import get_embeddings
from ai_agents.tools.vector_db import build_index, search_index

def match_customers(
    engine: str = "hf",
    combo: str = "address"
) -> pd.DataFrame:
    """
    For every CRM customer, find the top-1 matching E-com customer
    based on the chosen embedding engine & text combo.
    Returns a DataFrame with columns:
      - crm_idx, ecom_idx: integer row indices
      - crm_id, ecom_id: their customer_id values
    """
    # 1) load & clean
    crm, ecom = load_and_clean_all()

    # 2) prepare text (address-only)
    texts_crm = prep_address(crm)
    texts_ecom = prep_address(ecom)

    # 3) embed both lists
    embs_crm = get_embeddings(texts_crm, engine=engine)
    embs_ecom = get_embeddings(texts_ecom, engine=engine)

    # 4) build FAISS index on E-com embeddings
    index = build_index(embs_ecom)

    # 5) for each CRM embedding, query top-1 in E-com
    records = []
    for i, emb in enumerate(embs_crm):
        _, idxs = search_index(index, emb, k=1)
        j = idxs[0]
        records.append({
            "crm_idx": i,
            "ecom_idx": j,
            "crm_id": crm.loc[i, "customer_id"],
            "ecom_id": ecom.loc[j, "customer_id"],
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    # Run as a script: print first 10 matches and save CSV
    df_matches = match_customers(engine="hf", combo="address")
    print("First 10 customer matches:")
    print(df_matches.head(10))
    df_matches.to_csv("data/customer_matches.csv", index=False)
    print("→ Saved to data/customer_matches.csv")
