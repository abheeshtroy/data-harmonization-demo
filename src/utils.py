# src/utils.py
import os
from dotenv import load_dotenv

def load_env() -> str:
    """
    Load environment variables from .env and return the OPENAI_API_KEY.
    Raises RuntimeError if the key is missing.
    """
    load_dotenv()  # reads .env in the current working directory
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    return key
