"""
Dijkstra Agent Solver.
Greedy agent that always moves toward the goal using shortest path.
Only uses pawn moves (actions 0-11), never places walls.
"""

from typing import Optional, Dict, Any
import numpy as np
from solvers.base import BaseAgent
from solvers.registry import register_solver


class DijkstraAgent(BaseAgent):
    """
    Greedy agent that always moves toward the goal using shortest path.
    Only uses pawn moves (0-11), never places walls.
    
    This agent:
    1. Finds valid pawn moves from the action mask (actions 0-11)
    2. For each valid move, calculates the resulting distance to the goal
    3. Picks the move that minimizes distance
    """

    CONFIG_SCHEMA = {
        "player": 2,
        "seed": None,
    }

    def __init__(self, player: int = 2, seed: Optional[int] = None):
        """
        Args:
            player: Which player this agent is (1 or 2)
            seed: Random seed for tie-breaking
        """
        self.player = player
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Movement deltas for actions 0-11
        self.deltas = {
            0: (-1, 0),   # North
            1: (0, 1),    # East
            2: (1, 0),    # South
            3: (0, -1),   # West
            4: (-2, 0),   # Jump North
            5: (0, 2),    # Jump East
            6: (2, 0),    # Jump South
            7: (0, -2),   # Jump West
            8: (-1, 1),   # Slide NE
            9: (-1, -1),  # Slide NW
            10: (1, 1),   # Slide SE
            11: (1, -1),  # Slide SW
        }

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # Perspective is normalized by the environment.
        # Agent always sees itself as primary player (Channel 0).
        pos_channel = observation[:, :, 0]
        dist_channel = observation[:, :, 4]  # Current player distance to goal

        # Find current position
        curr_pos = np.unravel_index(np.argmax(pos_channel), pos_channel.shape)
        curr_r, curr_c = curr_pos

        # Find valid pawn moves (actions 0-11)
        valid_pawn_actions = []
        for action_idx in range(12):
            if action_mask[action_idx] > 0:
                valid_pawn_actions.append(action_idx)

        if not valid_pawn_actions:
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) == 0:
                return 0
            return int(self.rng.choice(valid_actions))

        # Evaluate each pawn move by resulting distance
        best_actions = []
        best_distance = float('inf')

        for action_idx in valid_pawn_actions:
            dr, dc = self.deltas[action_idx]
            new_r, new_c = curr_r + dr, curr_c + dc

            # Check bounds
            if 0 <= new_r < 9 and 0 <= new_c < 9:
                distance = dist_channel[new_r, new_c]

                if distance < best_distance:
                    best_distance = distance
                    best_actions = [action_idx]
                elif distance == best_distance:
                    best_actions.append(action_idx)

        if best_actions:
            return int(self.rng.choice(best_actions))
        else:
            return int(self.rng.choice(valid_pawn_actions))

    def reset(self):
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "player": self.player,
            "seed": self.seed,
        }


# Auto-register solver
register_solver("dijkstra", DijkstraAgent)
