"""
Random Agent Solver.
Plays uniformly random legal moves from the provided action mask.
"""

from typing import Optional, Dict, Any
import numpy as np
from solvers.base import BaseAgent
from solvers.registry import register_solver


class RandomAgent(BaseAgent):
    """
    Agent that plays uniformly random legal moves.
    Useful as a baseline for evaluation.
    """

    CONFIG_SCHEMA = {
        "seed": None,
    }

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0
        return int(self.rng.choice(valid_actions))

    def reset(self):
        pass

    def get_config(self) -> Dict[str, Any]:
        return {"seed": self.seed}


# Auto-register solver
register_solver("random", RandomAgent)
