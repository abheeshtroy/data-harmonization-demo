from ai_agents.tools import etl, vector_db
from ai_agents.tools.embeddings import get_name_embeddings

def run():
    crm, ecom = etl.load_and_clean_all()
    # same pipeline but on crm["address"] & ecom["address"]
