# src/ai_agents/agents/openai_embed_agent.py

import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

# 1) Load your .env into os.environ
load_dotenv()  

# 2) Grab the key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")

# 3) Initialize the model with the key
_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=api_key
)

def embed_list(texts: list[str]) -> "np.ndarray":
    """
    Embed a list of strings using OpenAI's embedding API.
    Returns a (n_texts, D) NumPy array.
    """
    return _model.embed_documents(texts)
