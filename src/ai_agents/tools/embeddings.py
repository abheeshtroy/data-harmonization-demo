# src/ai_agents/tools/embeddings.py

import numpy as np
from ai_agents.agents.hf_embed_agent     import embed_list as _hf
from ai_agents.agents.ollama_embed_agent import embed_list as _oll

ENGINES = {
    "hf":    _hf,
    "ollama":_oll,
}

def get_embeddings(texts: list[str], engine: str = "hf") -> np.ndarray:
    """
    Embed a list of texts using the chosen engine.
    Returns a (N, D) float32 NumPy array.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'; choose from {list(ENGINES)}")

    raw = ENGINES[engine](texts)

    # if it’s already an ndarray
    if isinstance(raw, np.ndarray):
        return raw.astype("float32")

    # if it’s a list
    if isinstance(raw, list):
        if not raw:
            return np.zeros((0, 0), dtype="float32")
        first = raw[0]
        if isinstance(first, dict) and "embedding" in first:
            arr = np.array([d["embedding"] for d in raw], dtype="float32")
        else:
            arr = np.array(raw, dtype="float32")
        return arr

    # otherwise coerce
    return np.array(raw, dtype="float32")
