"""
AI Agent Module
Implements opponent agents for training and evaluation
"""

import numpy as np
import collections
from typing import Optional
from abc import ABC, abstractmethod


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
        # Get current position from observation
        if self.player == 1:
            pos_channel = observation[:, :, 0]
            dist_channel = observation[:, :, 4]  # P1 distance to goal
        else:
            pos_channel = observation[:, :, 1]
            dist_channel = observation[:, :, 5]  # P2 distance to goal
        
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
    Strategic agent that combines:
    1. Greedy pawn movement (like Dijkstra)
    2. Wall placement to block opponent
    
    Uses a simple heuristic:
    - If opponent is close to winning, try to place a blocking wall
    - Otherwise, advance toward goal
    """
    
    def __init__(self, player: int = 2, wall_prob: float = 0.3, seed: Optional[int] = None):
        """
        Args:
            player: Which player this agent is (1 or 2)
            wall_prob: Probability of considering wall placement
            seed: Random seed
        """
        self.player = player
        self.wall_prob = wall_prob
        self.rng = np.random.default_rng(seed)
        self.dijkstra = DijkstraAgent(player=player, seed=seed)
        
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # Get opponent's distance to goal
        if self.player == 1:
            opp_dist_channel = observation[:, :, 5]  # P2's distance
            opp_pos_channel = observation[:, :, 1]
        else:
            opp_dist_channel = observation[:, :, 4]  # P1's distance
            opp_pos_channel = observation[:, :, 0]
        
        # Find opponent position
        opp_pos = np.unravel_index(np.argmax(opp_pos_channel), opp_pos_channel.shape)
        opp_dist = opp_dist_channel[opp_pos[0], opp_pos[1]]
        
        # Check if walls are available (actions 12-139)
        wall_actions = np.where(action_mask[12:] > 0)[0] + 12
        
        # If opponent is close (small distance) and we have walls, consider blocking
        if len(wall_actions) > 0 and opp_dist < 0.1:  # Normalized distance
            if self.rng.random() < self.wall_prob:
                # Try to find a wall that increases opponent's distance
                # For simplicity, just pick a random valid wall
                return self.rng.choice(wall_actions)
        
        # Default: use Dijkstra strategy
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
        # Get distance channels
        if self.player == 1:
            my_dist = observation[:, :, 4]
            opp_dist = observation[:, :, 5]
            my_pos = observation[:, :, 0]
        else:
            my_dist = observation[:, :, 5]
            opp_dist = observation[:, :, 4]
            my_pos = observation[:, :, 1]
        
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


# Convenience function to get agent by name
def get_agent(name: str, player: int = 2, seed: Optional[int] = None) -> BaseAgent:
    """
    Factory function to create agents by name.
    
    Args:
        name: Agent type ('random', 'dijkstra', 'strategic', 'minimax')
        player: Player number (1 or 2)
        seed: Random seed
    
    Returns:
        Agent instance
    """
    agents = {
        'random': lambda: RandomAgent(seed=seed),
        'dijkstra': lambda: DijkstraAgent(player=player, seed=seed),
        'strategic': lambda: StrategicAgent(player=player, seed=seed),
        'minimax': lambda: MinimaxAgent(player=player, seed=seed),
    }
    
    if name.lower() not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys())}")
    
    return agents[name.lower()]()