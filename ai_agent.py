"""
AI Agent Module
Implements opponent agents for training and evaluation
"""

import numpy as np
import collections
from typing import Optional
from abc import ABC, abstractmethod
from sb3_contrib import MaskablePPO

from quoridor_game import QuoridorStateBuilder


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select an action given the current observation and legal action mask.
        
        Args:
            observation: The 9x9x6 game state tensor
            action_mask: Binary mask of shape (140,) where 1 = valid action
            
        Returns:
            Action index (0-139)
        """
        pass
    
    def reset(self):
        """Reset agent state between episodes."""
        pass


class RandomAgent(BaseAgent):
    """
    Agent that plays uniformly random legal moves.
    Useful as a baseline for evaluation.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        valid_actions = np.where(action_mask > 0)[0]
        return self.rng.choice(valid_actions)


class DijkstraAgent(BaseAgent):
    """
    Greedy agent that always moves toward the goal using shortest path.
    Only uses pawn moves (0-11), never places walls.
    
    This agent:
    1. Finds valid pawn moves from the action mask (actions 0-11)
    2. For each valid move, calculates the resulting distance to the goal
    3. Picks the move that minimizes distance
    """
    
    def __init__(self, player: int = 2, seed: Optional[int] = None):
        """
        Args:
            player: Which player this agent is (1 or 2)
            seed: Random seed for tie-breaking
        """
        self.player = player
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
        # Perspective is now normalized by the environment.
        # Agent always sees itself as the primary player (Channel 0).
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
            # No pawn moves available, pick any valid action
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) == 0:
                # No valid actions at all - this should trigger game termination
                # Return 0 as a fallback (the environment should handle this)
                return 0
            return self.rng.choice(valid_actions)

        # Evaluate each pawn move by resulting distance
        best_actions = []
        best_distance = float('inf')

        for action_idx in valid_pawn_actions:
            dr, dc = self.deltas[action_idx]
            new_r, new_c = curr_r + dr, curr_c + dc

            # Check bounds
            if 0 <= new_r < 9 and 0 <= new_c < 9:
                # Get distance from heatmap (lower is better)
                distance = dist_channel[new_r, new_c]

                if distance < best_distance:
                    best_distance = distance
                    best_actions = [action_idx]
                elif distance == best_distance:
                    best_actions.append(action_idx)

        if best_actions:
            return self.rng.choice(best_actions)
        else:
            return self.rng.choice(valid_pawn_actions)


class StrategicAgent(BaseAgent):
    """
    Strategic agent that:
    1. Checks if it can place a wall to increase the opponent's path length.
    2. If multiple good walls exist, picks one randomly.
    3. If no wall increases path length (or based on probability), moves greedily towards goal (Dijkstra).
    """
    
    def __init__(self, player: int = 2, wall_prob: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            player: Which player this agent is (1 or 2)
            wall_prob: Probability of placing a wall if a good one is found.
            seed: Random seed
        """
        self.player = player
        self.wall_prob = wall_prob
        self.rng = np.random.default_rng(seed)
        self.dijkstra = DijkstraAgent(player=player, seed=seed)
        self.state_builder = QuoridorStateBuilder()
        
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # Perspective is now normalized.
        # Agent is always "Player 1" (Channel 0), Opponent is "Player 2" (Channel 1).
        # Opponent's goal is row 0 from this perspective.
        opp_pos_channel = observation[:, :, 1]
        opp_goal_row = 0
            
        opp_pos = np.unravel_index(np.argmax(opp_pos_channel), opp_pos_channel.shape)
        
        # 2. Get Current Wall Masks (already numpy arrays)
        h_walls_mask = observation[:, :, 2].astype(np.int8)
        v_walls_mask = observation[:, :, 3].astype(np.int8)

        # 3. Calculate Current Opponent Path Length
        base_heatmap = self.state_builder.get_shortest_path_heatmap(
            opp_pos, opp_goal_row, h_walls_mask, v_walls_mask
        )

        base_dist = base_heatmap[opp_pos]
        
        # 4. Evaluate Valid Wall Actions
        # Wall actions are 12 to 139
        wall_actions = np.where(action_mask[12:] > 0)[0] + 12
        
        best_wall_actions = []
        max_dist = base_dist
        
        # Only simulate if we decide to potentially place a wall
        # (We check wall_prob later? The prompt said "rather then placing a random wall it places ANY wall that makes opponent path longer")
        # So we should ALWAYS look for such a wall first.
        
        if len(wall_actions) > 0:
            for w_action in wall_actions:
                # Decode wall
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
                    
                # Calculate new distance
                new_heatmap = self.state_builder.get_shortest_path_heatmap(
                    opp_pos, opp_goal_row, temp_h, temp_v
                )
                new_dist = new_heatmap[opp_pos]
                
                if new_dist > max_dist:
                    max_dist = new_dist
                    best_wall_actions = [w_action]
                elif new_dist == max_dist and new_dist > base_dist:
                    # Keep track of all walls that maximize the distance equally
                    best_wall_actions.append(w_action)
        
        # 5. Select Action
        if best_wall_actions:
            # We found a wall that makes the path longer!
            # Use wall_prob to decide if we actually do it (to add some stochasticity/conservativeness?)
            # The prompt implies: "places any wall that makes opponent path longer".
            # It didn't explicitly say "always", but it replaced the "random wall" logic.
            # I'll stick to the existing wall_prob to determine IF we want to place a wall vs move.
            # BUT, if we DO place a wall, it must be one of the best ones.
            
            # However, the previous logic was: "If opp close AND wall_prob, place random wall".
            # New logic: "Place wall that makes path longer".
            
            # Let's say: If we found a wall that increases path, we prioritize it based on wall_prob.
            if self.rng.random() < self.wall_prob:
                 return self.rng.choice(best_wall_actions)
        
        # Fallback: Move towards goal (Dijkstra)
        return self.dijkstra.select_action(observation, action_mask)


class MinimaxAgent(BaseAgent):
    """
    Agent using minimax with alpha-beta pruning.
    Evaluates positions based on distance difference between players.
    
    Note: Can be slow for deep searches due to large branching factor.
    """
    
    def __init__(self, player: int = 2, depth: int = 2, seed: Optional[int] = None):
        """
        Args:
            player: Which player this agent is (1 or 2)
            depth: Search depth (2-3 recommended due to complexity)
            seed: Random seed for tie-breaking
        """
        self.player = player
        self.depth = depth
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """Use minimax to select best action."""
        valid_actions = np.where(action_mask > 0)[0]
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # For efficiency, limit to pawn moves if too many wall options
        pawn_actions = [a for a in valid_actions if a < 12]
        
        if len(pawn_actions) > 0:
            # Evaluate pawn moves using heuristic
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
        
        # Fallback to random valid action
        return self.rng.choice(valid_actions)
    
    def _evaluate_action(self, observation: np.ndarray, action: int) -> float:
        """
        Evaluate an action based on resulting position.
        Higher score = better for this player.
        """
        # Perspective is now normalized.
        my_dist = observation[:, :, 4]
        opp_dist = observation[:, :, 5]
        my_pos = observation[:, :, 0]
        
        # Current position
        curr_pos = np.unravel_index(np.argmax(my_pos), my_pos.shape)
        
        # Movement deltas
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
                # Score: how much closer we get to goal
                curr_dist = my_dist[curr_pos[0], curr_pos[1]]
                new_dist = my_dist[new_r, new_c]
                return curr_dist - new_dist  # Positive if we're getting closer
        
        return 0.0


class RLAgent(BaseAgent):
    """
    Agent powered by a trained RL model (MaskablePPO).
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
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=self.deterministic)
        if isinstance(action, np.ndarray):
            return action.item()
        return int(action)


# Convenience function to get agent by name
def get_agent(name: str, player: int = 2, seed: Optional[int] = None, model_path: Optional[str] = None) -> BaseAgent:
    """
    Factory function to create agents by name.
    
    Args:
        name: Agent type ('random', 'dijkstra', 'strategic', 'minimax', 'rl')
        player: Player number (1 or 2)
        seed: Random seed
        model_path: Path to model file (required for 'rl' agent)
    
    Returns:
        Agent instance
    """
    name = name.lower()
    
    if name == 'rl':
        if not model_path:
            raise ValueError("model_path is required for RL agent")
        return RLAgent(model_path, player=player, seed=seed)

    rng = np.random.default_rng(seed) 
    agents = {
        'mixed': lambda: DijkstraAgent(player, seed=seed) if rng.random() < 0.8 else StrategicAgent(player, seed=seed),
        'random': lambda: RandomAgent(seed=seed),
        'dijkstra': lambda: DijkstraAgent(player=player, seed=seed),
        'strategic': lambda: StrategicAgent(player=player, seed=seed),
        'minimax': lambda: MinimaxAgent(player=player, seed=seed),
    }

    
    if name not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys()) + ['rl']}")
    
    return agents[name]()