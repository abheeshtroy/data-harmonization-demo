# src/ai_agents/agents/pipeline_agent.py

import pandas as pd
from pathlib import Path
from ai_agents.agents.config_parser_agent import parse
from ai_agents.agents.validation_agent import ValidationAgent
from ai_agents.agents.matching_agent   import MatchingAgent

class PipelineAgent:
    """
    Orchestrates parsing a user instruction, picking the right agent
    (ValidationAgent for k>1, MatchingAgent for k=1), running it,
    and persisting the results.
    """
    def __init__(self, instruction: str):
        cfg = parse(instruction)
        self.combos = cfg["combos"]   # list[str]
        self.engine = cfg["engine"]   # "hf", "mini", or "ollama"
        self.k      = cfg["k"]        # how many neighbors
        self.rerank = cfg["rerank"]   # bool
        self.out    = cfg["out"]      # filepath to write CSV

    def run(self) -> pd.DataFrame:
        # Select agent based on k
        if self.k > 1:
            agent = ValidationAgent(
                engine=self.engine,
                combo=self.combos,
                k=self.k,
                rerank=self.rerank,
            )
        else:
            agent = MatchingAgent(
                engine=self.engine,
                combo=self.combos,
                k=self.k,
                rerank=self.rerank,
            )

        # Run the chosen agent
        df = agent.run()

        # Persist to CSV, creating parent dir if needed
        out_path = Path(self.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"→ Saved results to {out_path}")

        return df
