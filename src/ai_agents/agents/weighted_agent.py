import pandas as pd
from ai_agents.agents.ingestion_agent import load_crm, load_ecom
from ai_agents.agents.prep_agent    import prep_texts
from ai_agents.tools.embeddings      import get_embeddings
from ai_agents.tools.vector_db       import build_index, search_index

class WeightedComboAgent:
    """
    Given multiple field‐combos and per‐combo weights,
    compute a weighted sum of embedding similarities for CRM→E-com matching.
    """
    def __init__(
        self,
        combos:    list[str],
        weights:   dict[str, float],
        engine:    str = "hf",
        k:         int = 5,
        rerank:    bool = False,
    ):
        self.combos  = combos
        self.weights = weights
        self.engine  = engine
        self.k       = k
        self.rerank  = rerank

    def run(self) -> pd.DataFrame:
        # 1) load & clean
        crm, ecom = load_crm(), load_ecom()

        # 2) for each combo, prep texts and get embeddings
        embs_crm = {}
        embs_ecom = {}
        for combo in self.combos:
            t1, t2 = prep_texts(crm, ecom, combo=combo)
            embs_crm[combo]  = get_embeddings(t1.tolist(), engine=self.engine)
            embs_ecom[combo] = get_embeddings(t2.tolist(), engine=self.engine)

        # 3) build one FAISS index per combo
        idxs_per = {}
        scores_per = {}
        for combo, emb2 in embs_ecom.items():
            idx   = build_index(emb2)
            idxs_per[combo]   = []
            scores_per[combo] = []
            for vec in embs_crm[combo]:
                sc, ix = search_index(idx, vec, k=self.k)
                idxs_per[combo].append(ix)
                scores_per[combo].append(sc)

        # 4) for each CRM row, compute weighted score per candidate
        records = []
        n = len(crm)
        for i in range(n):
            # accumulate weighted similarity across combos for each candidate pos
            agg_scores = {}
            for combo in self.combos:
                w = self.weights.get(combo, 0)
                scs = scores_per[combo][i]
                ixs = idxs_per[combo][i]
                for pos, sc in zip(ixs, scs):
                    agg_scores.setdefault(pos, 0.0)
                    agg_scores[pos] += w * sc

            # pick top‐k overall by agg_score
            sorted_pos = sorted(agg_scores, key=lambda p: agg_scores[p], reverse=True)[: self.k]

            # optional fuzzy re‐rank (if you still want)
            if self.rerank and "name" in crm.columns:
                from ai_agents.tools.fuzzy_rerank import rerank_by_name
                crm_name = crm.loc[i, "name"]
                sorted_pos = rerank_by_name(crm_name, ecom, sorted_pos)

            # record only the top‐1 match
            top = sorted_pos[0]
            records.append({
                "crm_id":      crm.loc[i, "id"],
                "ecom_id":     ecom.loc[top, "id"],
                "match_score": float(agg_scores[top]),
                "engine":      self.engine,
                "combos":      ",".join(self.combos),
                "rerank":      self.rerank,
            })

        return pd.DataFrame(records)
