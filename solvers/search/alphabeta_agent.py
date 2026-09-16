"""
Alpha-Beta Negamax Agent Solver.
Depth-limited search with move ordering and alpha-beta pruning.
Includes telemetry tracking for RQ2 analysis (nodes_expanded, pruned_nodes, elapsed_ms).
"""

import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from solvers.base import BaseAgent
from solvers.registry import register_solver
from core.state import QuoridorStateBuilder


class AlphaBetaAgent(BaseAgent):
    """
    Alpha-Beta Negamax Agent.
    
    Optimizations:
    - Negamax formulation for standard tree traversal.
    - Move ordering: pawn moves sorted by shortest path distance first, followed by top-K wall placements.
    - Alpha-beta pruning.
    """

    CONFIG_SCHEMA = {
        "player": 2,
        "depth": 2,
        "wall_candidates_k": 10,
        "seed": None,
    }

    def __init__(
        self,
        player: int = 2,
        depth: int = 2,
        wall_candidates_k: int = 10,
        seed: Optional[int] = None,
    ):
        self.player = player
        self.depth = depth
        self.wall_candidates_k = wall_candidates_k
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.state_builder = QuoridorStateBuilder()

        # Telemetry
        self._nodes_expanded = 0
        self._pruned_nodes = 0
        self._last_elapsed_ms = 0.0

    @property
    def last_move_stats(self) -> dict:
        """Stats from the most recent select_action() call."""
        return {
            "nodes_expanded": self._nodes_expanded,
            "pruned_nodes": self._pruned_nodes,
            "elapsed_ms": self._last_elapsed_ms,
        }

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        start_time = time.perf_counter()
        self._nodes_expanded = 0
        self._pruned_nodes = 0

        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0
        if len(valid_actions) == 1:
            self._last_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return int(valid_actions[0])

        candidates = self._get_ordered_candidates(observation, action_mask)
        if not candidates:
            candidates = list(valid_actions)

        best_score = float("-inf")
        best_actions = []
        alpha = float("-inf")
        beta = float("inf")

        for action in candidates:
            next_obs, next_mask = self._simulate_step(observation, action_mask, action)
            score = -self._negamax(
                next_obs, next_mask, depth=self.depth - 1, alpha=-beta, beta=-alpha
            )

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

            alpha = max(alpha, best_score)

        self._last_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if best_actions:
            return int(self.rng.choice(best_actions))
        return int(self.rng.choice(valid_actions))

    def _negamax(
        self, obs: np.ndarray, mask: np.ndarray, depth: int, alpha: float, beta: float
    ) -> float:
        self._nodes_expanded += 1

        my_dist = self._get_dist(obs, 0)
        opp_dist = self._get_dist(obs, 1)

        if my_dist == 0:
            return 1000.0
        if opp_dist == 0:
            return -1000.0
        if depth <= 0:
            return self._evaluate_state(obs)

        candidates = self._get_ordered_candidates(obs, mask)
        if not candidates:
            return self._evaluate_state(obs)

        value = float("-inf")
        for i, action in enumerate(candidates):
            next_obs, next_mask = self._simulate_step(obs, mask, action)
            score = -self._negamax(
                next_obs, next_mask, depth - 1, -beta, -alpha
            )

            value = max(value, score)
            alpha = max(alpha, value)

            if alpha >= beta:
                # Count remaining candidates as pruned nodes
                self._pruned_nodes += len(candidates) - (i + 1)
                break

        return value

    def _evaluate_state(self, obs: np.ndarray) -> float:
        my_dist = self._get_dist(obs, 0)
        opp_dist = self._get_dist(obs, 1)
        return float(opp_dist - my_dist)

    def _get_dist(self, obs: np.ndarray, player_idx: int) -> int:
        channel_idx = 4 if player_idx == 0 else 5
        pos_channel = obs[:, :, player_idx]
        if not np.any(pos_channel > 0):
            return 999
        pos = np.unravel_index(np.argmax(pos_channel), pos_channel.shape)
        dist_norm = obs[pos[0], pos[1], channel_idx]
        dist = int(round(dist_norm * self.state_builder.max_dist))
        if dist >= int(self.state_builder.max_dist):
            return 999
        return dist

    def _get_ordered_candidates(
        self, obs: np.ndarray, mask: np.ndarray
    ) -> List[int]:
        """Move ordering: pawn moves sorted by resulting distance first, then sorted walls."""
        valid_actions = np.where(mask > 0)[0]
        pawn_moves = [a for a in valid_actions if a < 12]
        wall_moves = [a for a in valid_actions if a >= 12]

        # Order pawn moves by resulting distance
        curr_pos = np.unravel_index(np.argmax(obs[:, :, 0]), obs[:, :, 0].shape)
        dist_channel = obs[:, :, 4]

        deltas = {
            0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1),
            4: (-2, 0), 5: (0, 2), 6: (2, 0), 7: (0, -2),
            8: (-1, 1), 9: (-1, -1), 10: (1, 1), 11: (1, -1),
        }

        pawn_tuples = []
        for p_act in pawn_moves:
            dr, dc = deltas[p_act]
            nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
            d_val = dist_channel[nr, nc] if 0 <= nr < 9 and 0 <= nc < 9 else 999
            pawn_tuples.append((d_val, p_act))
        pawn_tuples.sort(key=lambda x: x[0])
        ordered_pawns = [p for _, p in pawn_tuples]

        if not wall_moves or self.wall_candidates_k <= 0:
            return ordered_pawns if ordered_pawns else list(valid_actions)

        # Wall candidate ordering
        opp_pos = np.unravel_index(np.argmax(obs[:, :, 1]), obs[:, :, 1].shape)
        opp_goal_row = 0
        h_mask = obs[:, :, 2].astype(np.int8)
        v_mask = obs[:, :, 3].astype(np.int8)

        base_hm = self.state_builder.get_shortest_path_heatmap(
            opp_pos, opp_goal_row, h_mask, v_mask
        )
        base_dist = base_hm[opp_pos]

        scored_walls = []
        for w_act in wall_moves:
            idx = w_act - 12
            is_h = idx < 64
            local_idx = idx if is_h else idx - 64
            r, c = local_idx // 8, local_idx % 8

            temp_h = h_mask.copy()
            temp_v = v_mask.copy()
            if is_h:
                temp_h[r, c] = 1
            else:
                temp_v[r, c] = 1

            new_hm = self.state_builder.get_shortest_path_heatmap(
                opp_pos, opp_goal_row, temp_h, temp_v
            )
            delta_l = new_hm[opp_pos] - base_dist
            if delta_l >= 1:
                scored_walls.append((delta_l, w_act))

        scored_walls.sort(key=lambda x: x[0], reverse=True)
        selected_walls = [w for _, w in scored_walls[: self.wall_candidates_k]]

        return ordered_pawns + selected_walls

    def _simulate_step(
        self, obs: np.ndarray, mask: np.ndarray, action: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        next_obs = obs.copy()

        if action < 12:
            deltas = {
                0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1),
                4: (-2, 0), 5: (0, 2), 6: (2, 0), 7: (0, -2),
                8: (-1, 1), 9: (-1, -1), 10: (1, 1), 11: (1, -1),
            }
            if action in deltas:
                dr, dc = deltas[action]
                curr = np.unravel_index(np.argmax(obs[:, :, 0]), obs[:, :, 0].shape)
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    next_obs[:, :, 0] = 0
                    next_obs[nr, nc, 0] = 1
        else:
            idx = action - 12
            is_h = idx < 64
            local_idx = idx if is_h else idx - 64
            r, c = local_idx // 8, local_idx % 8
            if is_h:
                next_obs[r, c, 2] = 1
            else:
                next_obs[r, c, 3] = 1

        my_pos = np.unravel_index(np.argmax(next_obs[:, :, 0]), next_obs[:, :, 0].shape)
        opp_pos = np.unravel_index(np.argmax(next_obs[:, :, 1]), next_obs[:, :, 1].shape)
        h_mask = next_obs[:, :, 2].astype(np.int8)
        v_mask = next_obs[:, :, 3].astype(np.int8)

        my_hm = self.state_builder.get_shortest_path_heatmap(my_pos, 8, h_mask, v_mask)
        opp_hm = self.state_builder.get_shortest_path_heatmap(opp_pos, 0, h_mask, v_mask)
        next_obs[:, :, 4] = my_hm / self.state_builder.max_dist
        next_obs[:, :, 5] = opp_hm / self.state_builder.max_dist

        next_mask = mask.copy()
        return next_obs, next_mask

    def reset(self):
        self._nodes_expanded = 0
        self._pruned_nodes = 0
        self._last_elapsed_ms = 0.0

    def get_config(self) -> Dict[str, Any]:
        return {
            "player": self.player,
            "depth": self.depth,
            "wall_candidates_k": self.wall_candidates_k,
            "seed": self.seed,
        }


# Auto-register solver
register_solver("alphabeta", AlphaBetaAgent)
