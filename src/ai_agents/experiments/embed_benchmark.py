# src/ai_agents/experiments/embed_benchmark.py

import pandas as pd
from ai_agents.tools.etl        import load_and_clean_all
from ai_agents.tools.prep_tool import prepare
from ai_agents.tools.embeddings import ENGINES, get_embeddings
from ai_agents.tools.vector_db  import build_index, search_index
from ai_agents.experiments.top5_accuracy import compute_mrr, compute_top5_accuracy
from ai_agents.tools.fuzzy_rerank    import rerank_by_name

def run_benchmark(engine: str, combo: str):
    # 1) load & clean
    crm, ecom = load_and_clean_all()
    ecom_names = ecom["name"].tolist()

    # 2) embed
    if combo != "weighted":
        texts_crm, texts_ecom = prepare(crm, ecom, combo)
        embs_crm = get_embeddings(texts_crm.tolist(),  engine=engine)
        embs_ecom = get_embeddings(texts_ecom.tolist(), engine=engine)

    else:
        # weighted combo: 30% name, 30% email, 20% phone, 20% address
        w = {"name": .3, "email": .3, "phone": .2, "address": .2}
        # get each field’s texts
        name_crm,  name_ecom  = prepare(crm,  ecom,  "name")
        email_crm, email_ecom = prepare(crm,  ecom,  "email")
        phone_crm, phone_ecom = prepare(crm,  ecom,  "phone")
        addr_crm,  addr_ecom  = prepare(crm,  ecom,  "address")
        # embed each separately
        emb_name_crm  = get_embeddings(name_crm.tolist(),  engine=engine)
        emb_name_ecom = get_embeddings(name_ecom.tolist(),  engine=engine)
        emb_email_crm  = get_embeddings(email_crm.tolist(), engine=engine)
        emb_email_ecom = get_embeddings(email_ecom.tolist(), engine=engine)
        emb_phone_crm  = get_embeddings(phone_crm.tolist(), engine=engine)
        emb_phone_ecom = get_embeddings(phone_ecom.tolist(), engine=engine)
        emb_addr_crm   = get_embeddings(addr_crm.tolist(),   engine=engine)
        emb_addr_ecom  = get_embeddings(addr_ecom.tolist(),   engine=engine)
        # weighted sum
        embs_crm = (
            w["name"]*emb_name_crm  +
            w["email"]*emb_email_crm+
            w["phone"]*emb_phone_crm+
            w["address"]*emb_addr_crm
        )
        embs_ecom = (
            w["name"]*emb_name_ecom  +
            w["email"]*emb_email_ecom+
            w["phone"]*emb_phone_ecom+
            w["address"]*emb_addr_ecom
        )

    # 3) build FAISS index
    index = build_index(embs_ecom)

    # 4) for each CRM embedding: retrieve top-5, fuzzy re-rank, record rank
    ranks = []
    for i, emb in enumerate(embs_crm):
        _, idxs = search_index(index, emb, k=5)
        # fuzzy re-rank those 5 by name similarity
        crm_name = crm.loc[i, "name"]
        reranked = rerank_by_name(crm_name, ecom_names, idxs.tolist())
        # compute final 1-based rank
        rank = reranked.index(i) + 1 if i in reranked else 6
        ranks.append(rank)

    # 5) compute metrics
    mrr  = compute_mrr(ranks)
    top5 = compute_top5_accuracy(ranks)
    print(f"{engine:>6} | {combo:<8} | MRR={mrr:.3f} | Top-5={top5:.3f}")

if __name__ == "__main__":
    COMBOS  = ["all", "name", "email", "phone", "address", "weighted"]

    for engine in ENGINES:
        for combo in COMBOS:
            run_benchmark(engine, combo)
