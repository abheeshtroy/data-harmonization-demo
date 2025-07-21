# src/ai_agents/tools/fuzzy_rerank.py
from rapidfuzz import fuzz

def rerank_by_name(crm_names, ecom_names, candidate_idxs):
    """
    Given a CRM name string and list of ecom name strings at candidate_idxs,
    compute fuzzy similarity and return the candidate index with max score.
    Returns the re-ranked list of indices sorted by descending score.
    """
    scores = []
    for idx in candidate_idxs:
        score = fuzz.token_sort_ratio(crm_names, ecom_names[idx]) / 100.0
        scores.append((idx, score))
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores]
