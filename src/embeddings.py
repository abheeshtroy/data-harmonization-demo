# src/embeddings.py
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

def get_name_embeddings(names: list[str]) -> np.ndarray:
    """
    Given a list of listing names, return an (N x D) float32 array of embeddings.
    """
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = [model.embed_query(name) for name in names]
    return np.array(vectors, dtype="float32")
