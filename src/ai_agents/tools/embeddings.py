import numpy as np
from ai_agents.agents.hf_embed_agent     import embed_list as _hf
from ai_agents.agents.ollama_embed_agent import embed_list as _oll
from sentence_transformers import SentenceTransformer

# instantiate a pure sentence-transformers MiniLM model
_st_model = SentenceTransformer("all-MiniLM-L6-v2")

def _mini(texts: list[str]):
    # returns List[List[float]]
    return _st_model.encode(texts).tolist()

ENGINES = {
    "hf":    _hf,
    "mini":  _mini,
    "ollama":_oll,
}

def get_embeddings(texts: list[str], engine: str = "hf") -> np.ndarray:
    """
    Embed a list of texts using the chosen engine.
    Returns an (N, D) float32 NumPy array.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'; choose from {list(ENGINES)}")

    raw = ENGINES[engine](texts)

    # if it’s already an ndarray
    if isinstance(raw, np.ndarray):
        return raw.astype("float32")

    # if it’s a list of dicts with an "embedding" key
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "embedding" in raw[0]:
        arr = np.array([d["embedding"] for d in raw], dtype="float32")
        return arr

    # if it’s a list of lists or list of floats
    arr = np.array(raw, dtype="float32")
    return arr
