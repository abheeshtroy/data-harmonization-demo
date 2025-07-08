# src/ai_agents/agents/ollama_embed_agent.py

import numpy as np
from langchain_community.embeddings import OllamaEmbeddings

# point this at the same model you’ve already pulled:
_model = OllamaEmbeddings(model="all-minilm:latest")

def embed_list(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings via Ollama, using LangChain’s OllamaEmbeddings.
    Returns a (N, D) float32 NumPy array.
    """
    embs = _model.embed_documents(texts)  # returns List[List[float]]
    return np.array(embs, dtype="float32")
