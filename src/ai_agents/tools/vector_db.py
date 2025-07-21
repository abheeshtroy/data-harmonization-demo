# src/ai_agents/tools/vector_db.py

import numpy as np
import faiss

def build_index(
    embeddings: list | np.ndarray | dict,
    index_path: str | None = None
) -> faiss.IndexFlat:
    """
    Build a FAISS index using cosine similarity (inner product on normalized vectors).
    Accepts:
      - a NumPy array of shape (N, D)
      - a list of float‐lists: [[…], […], …]
      - a list of dicts with key 'embedding'
    If `index_path` is given, writes the index to that file.
    Returns:
      the populated IndexFlatIP.
    """
    # 1) Convert incoming embeddings into a (N, D) float32 array
    if isinstance(embeddings, list):
        first = embeddings[0] if embeddings else []
        if isinstance(first, dict) and "embedding" in first:
            arr = np.array([item["embedding"] for item in embeddings], dtype="float32")
        else:
            arr = np.array(embeddings, dtype="float32")
    else:
        arr = embeddings.astype("float32")

    # 2) Normalize each vector to unit length
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.clip(norms, 1e-12, None)

    # 3) Build FAISS inner-product index
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    # 4) Optionally save
    if index_path:
        faiss.write_index(index, index_path)

    return index

def load_index(index_path: str) -> faiss.IndexFlat:
    """
    Load and return a FAISS index previously saved to `index_path`.
    """
    return faiss.read_index(index_path)

def search_index(
    index: faiss.IndexFlat,
    query_emb: list | np.ndarray | dict,
    k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query the FAISS index with cosine similarity.
    - index: the FAISS IndexFlatIP
    - query_emb: one embedding as list, array, or dict{'embedding': […]}
    - k: how many top neighbors to return
    Returns:
      (scores, indices) both 1D arrays of length k.
    """
    # 1) Normalize the query
    if isinstance(query_emb, dict) and "embedding" in query_emb:
        vec = np.array([query_emb["embedding"]], dtype="float32")
    elif isinstance(query_emb, list):
        vec = np.array([query_emb], dtype="float32")
    else:
        vec = np.asarray(query_emb, dtype="float32").reshape(1, -1)

    # unit-normalize
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec / np.clip(norm, 1e-12, None)

    # 2) Search
    scores, indices = index.search(vec, k)
    return scores[0], indices[0]
