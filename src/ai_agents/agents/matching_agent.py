# src/ai_agents/agents/matching_agent.py

from typing import Union, List
import pandas as pd
from ai_agents.agents.ingestion_agent import load_crm, load_ecom
from ai_agents.agents.prep_agent import prep_texts
from ai_agents.tools.embeddings import get_embeddings
from ai_agents.tools.vector_db import build_index, search_index
from ai_agents.tools.fuzzy_rerank import rerank_by_name

class MatchingAgent:
    """
    Embed & match CRM → E-com customers, pick the single best (rank=1).
    """
    def __init__(
        self,
        engine: str = "hf",
        combo:  Union[str, List[str]] = "all",
        k:      int = 1,
        rerank: bool = False,
    ):
        self.engine = engine
        self.combo  = combo
        self.k      = k
        self.rerank = rerank

    def run(self) -> pd.DataFrame:
        # 1) load & clean
        crm, ecom = load_crm(), load_ecom()

        # 2) prepare text fields
        texts_crm, texts_ecom = prep_texts(crm, ecom, combo=self.combo)

        # 3) embed
        embs_crm  = get_embeddings(texts_crm.tolist(), engine=self.engine)
        embs_ecom = get_embeddings(texts_ecom.tolist(), engine=self.engine)

        # 4) build FAISS index
        index = build_index(embs_ecom)

        # 5) match & (optional) rerank
        records = []
        for i, emb in enumerate(embs_crm):
            scores, idxs = search_index(index, emb, k=self.k)

            if self.rerank and len(idxs):
                idxs = rerank_by_name(crm.loc[i, "name"], ecom, idxs)

            top_j     = idxs[0] if len(idxs) else None
            top_score = float(scores[0]) if len(scores) else None

            records.append({
                "crm_id":      crm.loc[i, "id"],
                "ecom_id":     ecom.loc[top_j, "id"] if top_j is not None else None,
                "match_score": top_score,
                "engine":      self.engine,
                "combos":      self.combo,
                "rerank":      self.rerank,
            })

        return pd.DataFrame(records)

    def to_csv(self, path: str):
        df = self.run()
        df.to_csv(path, index=False)
        print(f"→ Saved matching results to {path}")
