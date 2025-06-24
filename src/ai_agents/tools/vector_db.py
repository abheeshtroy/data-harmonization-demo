# src/vector_db.py
import faiss
import numpy as np

def build_index(
    embeddings: np.ndarray,
    index_path: str = "name_index.faiss"
) -> faiss.Index:
    """
    Build and save a FAISS L2 index from embeddings.
    - embeddings: (N × D) float32 array
    - index_path: file path to write the index
    Returns the FAISS index object.
    """
    emb = embeddings.astype("float32")
    _, d = emb.shape

    # Create a flat (exact) L2 index
    index = faiss.IndexFlatL2(d)
    index.add(emb)

    # Persist to disk
    faiss.write_index(index, index_path)
    return index

def load_index(index_path: str = "name_index.faiss") -> faiss.Index:
    """
    Load and return a FAISS index saved on disk.
    """
    return faiss.read_index(index_path)

def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a FAISS index and a single query embedding,
    return the top-k (distances, indices).
    - query_embedding: (D,) or (1×D) float32 array
    """
    q = query_embedding.reshape(1, -1).astype("float32")
    distances, indices = index.search(q, k)
    return distances[0], indices[0]
