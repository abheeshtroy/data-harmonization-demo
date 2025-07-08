# src/ai_agents/agents/hf_embed_agent.py

from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def embed_list(texts: list[str]) -> "np.ndarray":
    """
    Embed a list of strings using HuggingFace Sentence-Transformers.
    Returns a 2D numpy array of shape (len(texts), D).
    """
    return _model.embed_documents(texts)
