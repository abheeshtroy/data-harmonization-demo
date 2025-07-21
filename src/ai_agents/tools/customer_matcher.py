import pandas as pd
from ai_agents.tools.etl        import load_and_clean_all
from ai_agents.tools.prep_tool  import prepare
from ai_agents.tools.embeddings import get_embeddings
from ai_agents.tools.vector_db  import build_index, search_index

def match_customers(engine: str="hf", combo: str="all") -> pd.DataFrame:
    crm, ecom = load_and_clean_all()
    texts_crm, texts_ecom = prepare(crm, ecom, combo)

    embs_crm = get_embeddings(texts_crm.tolist(), engine=engine)
    embs_ecom = get_embeddings(texts_ecom.tolist(), engine=engine)

    index = build_index(embs_ecom)

    records = []
    for i, emb in enumerate(embs_crm):
        _, idxs = search_index(index, emb, k=1)
        j = idxs[0]
        records.append({
            "crm_idx":  i,
            "ecom_idx": j,
            "crm_id":   crm.loc[i, "id"],
            "ecom_id":  ecom.loc[j, "id"],
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = match_customers(engine="hf", combo="all")
    print(df.head(10))
    df.to_csv("data/customer_matches.csv", index=False)
    print("→ Saved data/customer_matches.csv")
