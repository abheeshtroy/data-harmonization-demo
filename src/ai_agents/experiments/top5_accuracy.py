import numpy as np

def compute_mrr(ranks: list[int]) -> float:
    """Mean Reciprocal Rank given 1-based ranks (rank=6 if not found)."""
    rr = [1.0 / r for r in ranks]
    return float(np.mean(rr))

def compute_top5_accuracy(ranks: list[int]) -> float:
    """Fraction of true matches with rank ≤ 5."""
    return float(np.mean([1 if r <= 5 else 0 for r in ranks]))
