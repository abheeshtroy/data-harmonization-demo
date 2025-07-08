from ai_agents.tools import etl, vector_db
from ai_agents.tools.embeddings import get_name_embeddings

def run():
    crm, ecom = etl.load_and_clean_all()
    embs_crm  = get_name_embeddings(crm["name"].tolist())
    embs_ecom = get_name_embeddings(ecom["name"].tolist())
    # build index, search, compute Accuracy@1, MRR…
