"""
Backward compatibility shim for AI Agents.
Re-exports BaseAgent, heuristic agents, and factory functions from the solvers package.
"""

from typing import Optional
import numpy as np
from sb3_contrib import MaskablePPO

from solvers.base import BaseAgent
from solvers.heuristic.random_agent import RandomAgent
from solvers.heuristic.dijkstra_agent import DijkstraAgent
from solvers.heuristic.strategic_agent import StrategicAgent
from solvers.registry import get_solver, get_agent


# Temporary definitions for MinimaxAgent and RLAgent until moved in Stage 3A and Stage 3B
class MinimaxAgent(BaseAgent):
    """
    Agent using minimax with alpha-beta pruning.
    (To be migrated/rewritten in Stage 3A)
    """
    
    def __init__(self, player: int = 2, depth: int = 2, seed: Optional[int] = None):
        self.player = player
        self.depth = depth
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 1:
            return valid_actions[0]
        pawn_actions = [a for a in valid_actions if a < 12]
        if len(pawn_actions) > 0:
            best_actions = []
            best_score = float('-inf')
            for action in pawn_actions:
                score = self._evaluate_action(observation, action)
                if score > best_score:
                    best_score = score
                    best_actions = [action]
                elif score == best_score:
                    best_actions.append(action)
            return self.rng.choice(best_actions)
        return self.rng.choice(valid_actions)
    
    def _evaluate_action(self, observation: np.ndarray, action: int) -> float:
        my_dist = observation[:, :, 4]
        my_pos = observation[:, :, 0]
        curr_pos = np.unravel_index(np.argmax(my_pos), my_pos.shape)
        deltas = {
            0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1),
            4: (-2, 0), 5: (0, 2), 6: (2, 0), 7: (0, -2),
            8: (-1, 1), 9: (-1, -1), 10: (1, 1), 11: (1, -1),
        }
        if action in deltas:
            dr, dc = deltas[action]
            new_r = curr_pos[0] + dr
            new_c = curr_pos[1] + dc
            if 0 <= new_r < 9 and 0 <= new_c < 9:
                curr_dist = my_dist[curr_pos[0], curr_pos[1]]
                new_dist = my_dist[new_r, new_c]
                return curr_dist - new_dist
        return 0.0

    def get_config(self) -> dict:
        return {"player": self.player, "depth": self.depth, "seed": self.seed}


class RLAgent(BaseAgent):
    """
    Agent powered by a trained RL model (MaskablePPO).
    (To be migrated to solvers/rl/ppo_agent.py in Stage 3B)
    """

    def __init__(self, model_path: str, player: int = 2, seed: Optional[int] = None, deterministic: bool = False):
        try:
            self.model = MaskablePPO.load(model_path)
            self.model.set_random_seed(seed)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise e 
        self.deterministic = deterministic
        self.player = player
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=self.deterministic)
        if isinstance(action, np.ndarray):
            return action.item()
        return int(action)

    def get_config(self) -> dict:
        return {"player": self.player, "deterministic": self.deterministic, "seed": self.seed}


__all__ = [
    "BaseAgent",
    "RandomAgent",
    "DijkstraAgent",
    "StrategicAgent",
    "MinimaxAgent",
    "RLAgent",
    "get_agent",
]