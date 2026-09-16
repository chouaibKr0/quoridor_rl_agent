"""
PPO Agent Solver.
Wraps a trained MaskablePPO model into the BaseAgent interface.
"""

from typing import Optional, Dict, Any
import numpy as np
from sb3_contrib import MaskablePPO

from solvers.base import BaseAgent
from solvers.registry import register_solver


class PPOAgent(BaseAgent):
    """
    Agent powered by a trained RL model (MaskablePPO).
    """

    CONFIG_SCHEMA = {
        "model_path": None,
        "deterministic": False,
        "player": 2,
        "seed": None,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        deterministic: bool = False,
        player: int = 2,
        seed: Optional[int] = None,
    ):
        self.model_path = model_path
        self.deterministic = deterministic
        self.player = player
        self.seed = seed
        self.model = None

        if model_path:
            self.load_model(model_path, seed=seed)

    def load_model(self, model_path: str, seed: Optional[int] = None):
        """Load a trained MaskablePPO zip model."""
        self.model_path = model_path
        self.model = MaskablePPO.load(model_path)
        if seed is not None:
            self.model.set_random_seed(seed)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if self.model is None:
            # Fallback if no model loaded: uniform random legal action
            valid = np.where(action_mask > 0)[0]
            if len(valid) == 0:
                return 0
            return int(valid[0])

        action, _ = self.model.predict(
            observation, action_masks=action_mask, deterministic=self.deterministic
        )
        if isinstance(action, np.ndarray):
            return int(action.item())
        return int(action)

    def reset(self):
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "deterministic": self.deterministic,
            "player": self.player,
            "seed": self.seed,
        }


# Backward-compatibility alias
RLAgent = PPOAgent

# Auto-register solvers
register_solver("ppo", PPOAgent)
register_solver("rl", PPOAgent)
