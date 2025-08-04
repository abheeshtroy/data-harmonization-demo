import json
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

# 1) Instantiate your Ollama‐based LLM (must support text‐generation)
_llm = OllamaLLM(
    model="llama2:7b",
    temperature=0,
)

# 2) PromptTemplate: spelling out exactly the JSON keys we expect
_PROMPT = PromptTemplate(
    template="""
You are given a user instruction describing which field-combos, embedding engine,
top-K, whether to apply fuzzy re-ranking, any per-field weights, and the desired
output path.

Spell each JSON key exactly as follows:
- combos: list of any of ["all","name","email","phone","address"]
- engine: one of ["hf","mini","ollama"]
- k: integer number of neighbors to consider
- rerank: true or false
- out: string filepath to write results
- weights: OPTIONAL, either a string "name=0.7,address=0.3" or a JSON object
           mapping each combo to a float weight

Instruction:
\"\"\"
{instruction}
\"\"\"

Respond with only valid JSON, nothing else.
""",
    input_variables=["instruction"],
)

# 3) Build the chain
_chain = LLMChain(llm=_llm, prompt=_PROMPT)

def parse(instruction: str) -> dict:
    """
    Parse a free-form instruction into a dict with keys:
      - combos (list[str])
      - engine (str)
      - k (int)
      - rerank (bool)
      - out (str)
      - weights (dict[str,float], optional)
    """
    raw = _chain.run(instruction=instruction)
    data = json.loads(raw)

    # sometimes the model outputs "combs" by mistake
    if "combs" in data and "combos" not in data:
        data["combos"] = data.pop("combs")

    # normalize combos to a Python list
    combos = data.get("combos")
    if isinstance(combos, str):
        data["combos"] = [c.strip() for c in combos.split(",")]

    # parse weights if provided as a comma-sep string
    if "weights" in data and isinstance(data["weights"], str):
        wdict = {}
        for part in data["weights"].split(","):
            if "=" in part:
                field, val = [p.strip() for p in part.split("=", 1)]
                try:
                    wdict[field] = float(val)
                except ValueError:
                    raise ValueError(f"Invalid weight value for {field}: {val}")
        data["weights"] = wdict

    return data
