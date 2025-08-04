# src/ai_agents/agents/validation_agent.py

from typing import Union, List
import pandas as pd
from ai_agents.agents.ingestion_agent import load_crm, load_ecom
from ai_agents.agents.prep_agent import prep_texts
from ai_agents.tools.embeddings import get_embeddings
from ai_agents.tools.vector_db import build_index, search_index
from ai_agents.tools.fuzzy_rerank import rerank_by_name

class ValidationAgent:
    """
    Embed & match CRM → E-com customers for validation.
    Can optionally fuzzy-rerank the top-k by name similarity.
    """
    def __init__(
        self,
        engine: str = "hf",
        combo:  Union[str, List[str]] = "all",
        k:      int = 5,
        rerank: bool = False,
    ):
        self.engine = engine
        self.combo  = combo
        self.k      = k
        self.rerank = rerank

    def run(self) -> pd.DataFrame:
        # 1) load & clean
        crm  = load_crm()
        ecom = load_ecom()

        # 2) prepare the text fields (combo may be str or list[str])
        texts_crm, texts_ecom = prep_texts(crm, ecom, combo=self.combo)

        # 3) embed both
        emb_crm  = get_embeddings(texts_crm.tolist(), engine=self.engine)
        emb_ecom = get_embeddings(texts_ecom.tolist(),  engine=self.engine)

        # 4) build FAISS index on E-com
        index = build_index(emb_ecom)

        # 5) loop CRM, query top-k, optional rerank, assemble records
        records = []
        for i, q_emb in enumerate(emb_crm):
            scores, idxs = search_index(index, q_emb, k=self.k)

            if self.rerank and len(idxs):
                # rerank_by_name expects: (single_crm_name, ecom_df, idx_array)
                idxs = rerank_by_name(crm.loc[i, "name"], ecom, idxs)

            # record only the best match (rank=1 position in this list)
            top_idx   = idxs[0] if len(idxs) else None
            top_score = float(scores[0]) if len(scores) else None

            records.append({
                "crm_id": crm.loc[i, "id"],
                "ecom_id": ecom.loc[top_idx, "id"] if top_idx is not None else None,
                "rank":   1,
                "score":  top_score,
                "engine": self.engine,
                "combo":  self.combo,
                "rerank": self.rerank,
            })

        return pd.DataFrame(records)

    def to_csv(self, path: str):
        df = self.run()
        df.to_csv(path, index=False)
        print(f"→ Saved validation results to {path}")
