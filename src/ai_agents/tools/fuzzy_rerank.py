# src/ai_agents/tools/fuzzy_rerank.py

from rapidfuzz import fuzz

def rerank_by_name(
    crm_name: str,
    ecom_df,          # pandas.DataFrame with a 'name' column
    idxs: list[int]
) -> list[int]:
    """
    Given a single CRM name, the full E‐com DataFrame, and a list of candidate indices,
    compute rapidfuzz.token_sort_ratio for each, sort descending, and return
    the sorted list of indices.
    """
    # compute (idx, score) pairs
    scored = [
        (idx, fuzz.token_sort_ratio(crm_name, ecom_df.loc[idx, "name"]))
        for idx in idxs
    ]
    # sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored]
