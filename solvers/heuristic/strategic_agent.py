"""
Strategic Agent Solver.
Heuristic agent combining wall placements (to block opponent path) and greedy pawn movement.
"""

from typing import Optional, Dict, Any
import numpy as np
from solvers.base import BaseAgent
from solvers.registry import register_solver
from solvers.heuristic.dijkstra_agent import DijkstraAgent
from core.state import QuoridorStateBuilder


class StrategicAgent(BaseAgent):
    """
    Strategic agent that:
    1. Checks if it can place a wall to increase the opponent's path length.
    2. If multiple good walls exist, picks one randomly.
    3. If no wall increases path length (or based on probability), moves greedily towards goal (Dijkstra).
    """

    CONFIG_SCHEMA = {
        "player": 2,
        "wall_prob": 0.5,
        "seed": None,
    }

    def __init__(self, player: int = 2, wall_prob: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            player: Which player this agent is (1 or 2)
            wall_prob: Probability of placing a wall if a good one is found.
            seed: Random seed
        """
        self.player = player
        self.wall_prob = wall_prob
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dijkstra = DijkstraAgent(player=player, seed=seed)
        self.state_builder = QuoridorStateBuilder()

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # Perspective is normalized.
        # Agent is always "Player 1" (Channel 0), Opponent is "Player 2" (Channel 1).
        # Opponent's goal is row 0 from this perspective.
        opp_pos_channel = observation[:, :, 1]
        opp_goal_row = 0

        opp_pos = np.unravel_index(np.argmax(opp_pos_channel), opp_pos_channel.shape)

        # Get Current Wall Masks
        h_walls_mask = observation[:, :, 2].astype(np.int8)
        v_walls_mask = observation[:, :, 3].astype(np.int8)

        # Calculate Current Opponent Path Length
        base_heatmap = self.state_builder.get_shortest_path_heatmap(
            opp_pos, opp_goal_row, h_walls_mask, v_walls_mask
        )

        base_dist = base_heatmap[opp_pos]

        # Evaluate Valid Wall Actions (12 to 139)
        wall_actions = np.where(action_mask[12:] > 0)[0] + 12

        best_wall_actions = []
        max_dist = base_dist

        if len(wall_actions) > 0:
            for w_action in wall_actions:
                w_idx = w_action - 12
                is_h = w_idx < 64
                local_idx = w_idx if is_h else w_idx - 64
                r = local_idx // 8
                c = local_idx % 8

                temp_h = h_walls_mask.copy()
                temp_v = v_walls_mask.copy()
                if is_h:
                    temp_h[r, c] = 1
                else:
                    temp_v[r, c] = 1

                new_heatmap = self.state_builder.get_shortest_path_heatmap(
                    opp_pos, opp_goal_row, temp_h, temp_v
                )
                new_dist = new_heatmap[opp_pos]

                if new_dist > max_dist:
                    max_dist = new_dist
                    best_wall_actions = [w_action]
                elif new_dist == max_dist and new_dist > base_dist:
                    best_wall_actions.append(w_action)

        if best_wall_actions and (self.rng.random() < self.wall_prob):
            return int(self.rng.choice(best_wall_actions))

        return self.dijkstra.select_action(observation, action_mask)

    def reset(self):
        self.dijkstra.reset()

    def get_config(self) -> Dict[str, Any]:
        return {
            "player": self.player,
            "wall_prob": self.wall_prob,
            "seed": self.seed,
        }


# Auto-register solver
register_solver("strategic", StrategicAgent)
