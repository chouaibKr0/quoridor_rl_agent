"""
Pure Monte Carlo Tree Search (MCTS) Agent Solver.
Uses UCT selection and state cloning for rollout simulation.
Includes telemetry tracking for RQ2 budget measurement.
"""

import time
import math
from typing import Optional, Dict, Any, List
import numpy as np

from solvers.base import BaseAgent
from solvers.registry import register_solver
from core.game import QuoridorGame, flip_observation, flip_action, flip_mask
from solvers.heuristic.dijkstra_agent import DijkstraAgent


class MCTSNode:
    """Node in the MCTS search tree."""

    def __init__(self, game_state: QuoridorGame, parent: Optional["MCTSNode"] = None, action: Optional[int] = None):
        self.game = game_state.clone()
        self.parent = parent
        self.action = action
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[int, "MCTSNode"] = {}

        # Legal actions at this node
        mask = self.game.get_legal_moves()
        self.untried_actions = list(np.where(mask > 0)[0])

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.game.done

    def best_child(self, c_puct: float) -> "MCTSNode":
        best_score = float("-inf")
        best_nodes = []

        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                q = child.value_sum / child.visits
                u = c_puct * math.sqrt(math.log(self.visits) / child.visits)
                score = q + u

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        return best_nodes[0] if len(best_nodes) == 1 else best_nodes[0]


class MCTSAgent(BaseAgent):
    """
    Pure Monte Carlo Tree Search Agent.
    
    Phases per simulation:
    1. Selection (UCT)
    2. Expansion
    3. Simulation / Rollout (Random or Dijkstra)
    4. Backpropagation
    """

    CONFIG_SCHEMA = {
        "player": 2,
        "simulations": 100,
        "rollout_policy": "random",
        "c_puct": 1.414,
        "seed": None,
    }

    def __init__(
        self,
        player: int = 2,
        simulations: int = 100,
        rollout_policy: str = "random",
        c_puct: float = 1.414,
        seed: Optional[int] = None,
    ):
        self.player = player
        self.simulations = simulations
        self.rollout_policy = rollout_policy.lower()
        self.c_puct = c_puct
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dijkstra_agent = DijkstraAgent(player=player, seed=seed)

        # Telemetry
        self._simulations_run = 0
        self._nodes_expanded = 0
        self._last_elapsed_ms = 0.0

    @property
    def last_move_stats(self) -> dict:
        """Stats from the most recent select_action() call."""
        return {
            "simulations_run": self._simulations_run,
            "nodes_expanded": self._nodes_expanded,
            "elapsed_ms": self._last_elapsed_ms,
        }

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        start_time = time.perf_counter()
        self._simulations_run = 0
        self._nodes_expanded = 0

        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0
        if len(valid_actions) == 1:
            self._last_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return int(valid_actions[0])

        # Reconstruct game state instance from observation for root node
        root_game = self._reconstruct_game_from_obs(observation)
        root = MCTSNode(root_game)

        for _ in range(self.simulations):
            self._simulations_run += 1
            node = root

            # 1. Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c_puct)

            # 2. Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                action = node.untried_actions.pop()
                child_game = node.game.clone()
                child_game.step(action)
                child_node = MCTSNode(child_game, parent=node, action=action)
                node.children[action] = child_node
                node = child_node
                self._nodes_expanded += 1

            # 3. Simulation
            reward = self._rollout(node.game)

            # 4. Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += reward
                curr = curr.parent

        self._last_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Choose action with highest visit count
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

    def _rollout(self, game: QuoridorGame, max_depth: int = 40) -> float:
        """Run a simulation from current game state to terminal or max_depth."""
        sim_game = game.clone()
        depth = 0
        root_player = game.current_player

        while not sim_game.done and depth < max_depth:
            mask = sim_game.get_legal_moves()
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                break

            if self.rollout_policy == "dijkstra":
                obs = sim_game._get_observation()
                if sim_game.current_player != 1:
                    obs = flip_observation(obs)
                    mask = flip_mask(mask)
                    act = self.dijkstra_agent.select_action(obs, mask)
                    act = flip_action(act)
                else:
                    act = self.dijkstra_agent.select_action(obs, mask)
            else:
                act = int(self.rng.choice(valid))

            sim_game.step(act)
            depth += 1

        if sim_game.done:
            # Winner is player who reached destination
            if sim_game.p1_pos[0] == 8:
                winner = 1
            else:
                winner = 2
            return 1.0 if winner == root_player else -1.0
        
        # Non-terminal cutoff evaluation by relative distance
        obs = sim_game._get_observation()
        p1_dist = obs[sim_game.p1_pos[0], sim_game.p1_pos[1], 4]
        p2_dist = obs[sim_game.p2_pos[0], sim_game.p2_pos[1], 5]

        if root_player == 1:
            return float(p2_dist - p1_dist)
        else:
            return float(p1_dist - p2_dist)

    def _reconstruct_game_from_obs(self, obs: np.ndarray) -> QuoridorGame:
        """Helper to create a QuoridorGame matching current observation."""
        g = QuoridorGame()
        p1_pos = np.unravel_index(np.argmax(obs[:, :, 0]), (9, 9))
        p2_pos = np.unravel_index(np.argmax(obs[:, :, 1]), (9, 9))
        g.p1_pos = (int(p1_pos[0]), int(p1_pos[1]))
        g.p2_pos = (int(p2_pos[0]), int(p2_pos[1]))
        g.h_walls_mask = obs[:, :, 2].astype(np.int8)
        g.v_walls_mask = obs[:, :, 3].astype(np.int8)
        return g

    def reset(self):
        self._simulations_run = 0
        self._nodes_expanded = 0
        self._last_elapsed_ms = 0.0

    def get_config(self) -> Dict[str, Any]:
        return {
            "player": self.player,
            "simulations": self.simulations,
            "rollout_policy": self.rollout_policy,
            "c_puct": self.c_puct,
            "seed": self.seed,
        }


# Auto-register solver
register_solver("mcts", MCTSAgent)
