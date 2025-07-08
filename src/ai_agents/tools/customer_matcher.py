# src/ai_agents/tools/customer_matcher.py

from ai_agents.tools.etl        import load_crm, load_ecom, clean_crm, clean_ecom
from ai_agents.tools.embeddings import get_name_embeddings
from ai_agents.tools.vector_db  import build_index, search_index

def main():
    # 1. Load & clean both tables
    crm   = clean_crm(load_crm("data/dataset1_crm.csv"))
    ecom  = clean_ecom(load_ecom("data/dataset2_ecommerce.csv"))

    # 2. Extract the shared 'name' column and embed
    names_crm  = crm["name"].tolist()
    names_ecom = ecom["name"].tolist()
    embs_crm   = get_name_embeddings(names_crm)
    embs_ecom  = get_name_embeddings(names_ecom)

    # 3. Build a FAISS index over the E-com embeddings
    index = build_index(embs_ecom, index_path="ecom_index.faiss")

    # 4. For each CRM name, find and print the closest E-com match
    print("CRM name → E-com name (distance)")
    for i, emb in enumerate(embs_crm):
        dists, idxs = search_index(index, emb, k=1)
        crm_name  = crm.loc[i, "name"]
        ecom_name = ecom.loc[idxs[0], "name"]
        print(f"{crm_name!r} → {ecom_name!r} (dist={dists[0]:.3f})")

if __name__ == "__main__":
    main()
