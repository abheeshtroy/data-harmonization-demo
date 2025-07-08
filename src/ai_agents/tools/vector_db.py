# src/ai_agents/tools/vector_db.py

import numpy as np
import faiss


def build_index(
    embeddings: list | np.ndarray | dict,
    index_path: str | None = None
) -> faiss.IndexFlat:
    """
    Build a FAISS L2 index from embeddings.
    Accepts:
      - a NumPy array of shape (N, D)
      - a list of float‐lists: [[…], […], …]
      - a list of dicts with key 'embedding'
    If `index_path` is given, writes the index to that file.
    Returns:
      the populated IndexFlatL2.
    """
    # normalize into a (N, D) float32 array
    if isinstance(embeddings, list):
        first = embeddings[0] if embeddings else None
        # list of dicts?
        if isinstance(first, dict) and "embedding" in first:
            arr = np.array([item["embedding"] for item in embeddings], dtype="float32")
        else:
            arr = np.array(embeddings, dtype="float32")
    else:
        # assume numpy array
        arr = np.asarray(embeddings, dtype="float32")

    # build the index
    dim = arr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(arr)

    # optionally write to disk
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
    Query the FAISS index.
    - index: the FAISS IndexFlatL2
    - query_emb: one embedding as a list, numpy array, or dict{'embedding': […]}
    - k: how many nearest neighbors to return
    Returns:
      (distances, indices) both 1D arrays of length k.
    """
    # normalize to shape (1, D)
    if isinstance(query_emb, dict) and "embedding" in query_emb:
        vec = np.array([query_emb["embedding"]], dtype="float32")
    elif isinstance(query_emb, list):
        vec = np.array([query_emb], dtype="float32")
    else:
        vec = np.asarray(query_emb, dtype="float32").reshape(1, -1)

    distances, indices = index.search(vec, k)
    return distances[0], indices[0]
