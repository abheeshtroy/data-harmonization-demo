# src/ai_agents/experiments/embed_benchmark.py

import numpy as np
import pandas as pd
from ai_agents.tools.etl import get_ground_truth
from ai_agents.tools.embeddings import get_embeddings
from ai_agents.tools.vector_db import build_index, search_index

# Define embedding engines and field combinations to test
EMBED_ENGINES = ["hf", "ollama"]
FIELD_COMBOS = ["name", "address", "name_address"]


def prepare_texts(df: pd.DataFrame, combo: str):
    """
    Given the ground-truth DataFrame (with suffixes _crm and _ecom),
    return two lists of strings for CRM and E-com based on combo.
    """
    if combo == "name":
        texts1 = df["name_crm"].astype(str)
        texts2 = df["name_ecom"].astype(str)
    elif combo == "address":
        texts1 = df["address_crm"].astype(str)
        texts2 = df["address_ecom"].astype(str)
    elif combo == "name_address":
        texts1 = (df["name_crm"].astype(str) + "; " + df["address_crm"].astype(str))
        texts2 = (df["name_ecom"].astype(str) + "; " + df["address_ecom"].astype(str))
    else:
        raise ValueError(f"Unknown combo: {combo}")

    # Basic text normalization: lowercase and strip
    texts1 = texts1.str.lower().str.strip().tolist()
    texts2 = texts2.str.lower().str.strip().tolist()
    return texts1, texts2


def compute_mrr(texts1, texts2, engine: str) -> float:
    """
    Compute Mean Reciprocal Rank for a given engine on the two lists.
    """
    # 1) Embed lists
    embs1 = get_embeddings(texts1, engine=engine)
    embs2 = get_embeddings(texts2, engine=engine)

    # 2) Build FAISS index on E-com embeddings
    index = build_index(embs2)

    # 3) For each CRM embedding, query and compute reciprocal rank
    rr_total = 0.0
    for i, emb in enumerate(embs1):
        dists, idxs = search_index(index, emb, k=len(embs2))
        # find where ground-truth partner (same row index) appears
        ranks = np.where(idxs == i)[0]
        if ranks.size > 0:
            rr_total += 1.0 / (ranks[0] + 1)
        # else: contributes 0
    return rr_total / len(embs1)


def main():
    # 1) Get a small ground-truth set (first 20 matches)
    df_gt = get_ground_truth(n=20)

    results = []
    # 2) Loop engines and field combos
    for engine in EMBED_ENGINES:
        for combo in FIELD_COMBOS:
            texts1, texts2 = prepare_texts(df_gt, combo)
            mrr = compute_mrr(texts1, texts2, engine)
            results.append({
                "engine": engine,
                "field_combo": combo,
                "MRR": round(mrr, 3)
            })
            print(f"Engine={engine:<7} Combo={combo:<12} MRR={mrr:.3f}")

    # 3) Tabulate and save
    df_res = pd.DataFrame(results)
    print("\nSummary:")
    print(df_res)
    df_res.to_csv("data/embed_benchmark_results.csv", index=False)


if __name__ == "__main__":
    main()
