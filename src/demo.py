# src/demo.py

from src.utils       import load_env
from src.etl         import load_data, clean_data
from src.embeddings  import get_name_embeddings
from src.vector_db   import build_index, load_index, search_index
import numpy as np

def find_duplicates(
    embs: np.ndarray,
    threshold: float = 0.8,
    top_k: int = 10
) -> dict[int, set[int]]:
    """
    Returns a mapping from each index i to a set of duplicate indices.
    Uses a simple union-find approach.
    """
    n = embs.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build a temporary in-memory FAISS index
    idx = build_index(embs, index_path="temp.faiss")

    for i in range(n):
        dists, idxs = search_index(idx, embs[i], k=top_k)
        for dist, j in zip(dists, idxs):
            if i < j and dist <= threshold:
                union(i, j)

    # Aggregate clusters
    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, set()).add(i)

    # Only return clusters with more than 1 member
    return {r: grp for r, grp in clusters.items() if len(grp) > 1}

def main(
    csv_path: str = "data/AB_NYC_2019.csv",
    threshold: float = 0.8,
    top_k: int = 10,
    sample_n: int = 500
):
    load_env()
    df = clean_data(load_data(csv_path))
    # Sample a smaller subset for demonstration
    df_sample = df.head(sample_n).reset_index(drop=True)

    # 1. Get embeddings
    names = df_sample["name"].tolist()
    embs = get_name_embeddings(names)

    # 2. Find duplicates
    dup_clusters = find_duplicates(embs, threshold=threshold, top_k=top_k)
    print(f"Found {len(dup_clusters)} duplicate clusters (threshold={threshold})\n")

    # 3. Display each cluster
    for root, members in dup_clusters.items():
        print(f"Cluster rooted at {root}:")
        for idx in sorted(members):
            print(f"  - [{idx}] {df_sample.loc[idx, 'name']!r}")
        print()

if __name__ == "__main__":
    main()

