"""
Gymnasium Environment Wrapper for Quoridor
Provides a proper RL interface with action masking support
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from quoridor_game import QuoridorGame


class QuoridorEnv(gym.Env):
    """
    Gymnasium environment for Quoridor with action masking support.
    
    Observation Space: Box(0, 1, shape=(9, 9, 6), dtype=float32)
        - Channel 0: Player 1 position (one-hot)
        - Channel 1: Player 2 position (one-hot)
        - Channel 2: Horizontal walls
        - Channel 3: Vertical walls
        - Channel 4: Player 1 distance heatmap
        - Channel 5: Player 2 distance heatmap
    
    Action Space: Discrete(140)
        - 0-3: Step N, E, S, W
        - 4-7: Jump N, E, S, W (straight jump over opponent)
        - 8-11: Slide NE, NW, SE, SW (diagonal moves)
        - 12-75: Horizontal walls (64 positions)
        - 76-139: Vertical walls (64 positions)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        opponent: Optional[Any] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
    ):
        """
        Initialize the Quoridor environment.
        
        Args:
            opponent: Agent to play as player 2. If None, environment works
                     in self-play mode (returns observations for alternating players).
            render_mode: 'human' for visual, 'ansi' for text output.
            max_steps: Maximum steps before truncation.
        """
        super().__init__()
        
        self.game = QuoridorGame()
        self.opponent = opponent
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.steps = 0
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9, 9, 6),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(140)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.game.reset()
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        For self-play (opponent=None): alternates between players.
        With opponent: agent is always player 1, opponent responds immediately.
        """
        self.steps += 1
        
        # Player 1's action
        obs, reward, done, info = self.game.step(action)
        
        if done:
            return obs, reward, True, False, info
        
        # If we have an opponent and it's now player 2's turn
        if self.opponent is not None and self.game.current_player == 2:
            # Get opponent's action
            opp_mask = self.game.get_legal_moves()
            opp_action = self.opponent.select_action(obs, opp_mask)
            
            # Execute opponent's action
            obs, opp_reward, done, info = self.game.step(opp_action)
            
            # Reward from agent's perspective (opponent winning is bad)
            if done and self.game.p2_pos[0] == 0:
                reward -= 1.0  # Opponent won
        
        # Check truncation
        truncated = self.steps >= self.max_steps
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return self.game._get_observation()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        return {
            "current_player": self.game.current_player,
            "p1_walls_left": self.game.p1_walls_left,
            "p2_walls_left": self.game.p2_walls_left,
            "p1_pos": self.game.p1_pos,
            "p2_pos": self.game.p2_pos,
            "steps": self.steps,
        }
    
    def action_masks(self) -> np.ndarray:
        """
        Return valid action mask for MaskablePPO.
        
        Returns:
            Binary mask of shape (140,) where 1 = valid action.
        """
        return self.game.get_legal_moves().astype(bool)
    
    def render(self):
        """Render the current game state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_ansi()
    
    def _render_ansi(self) -> str:
        """Render game state as ASCII."""
        board = [["." for _ in range(9)] for _ in range(9)]
        
        # Place players
        p1_r, p1_c = self.game.p1_pos
        p2_r, p2_c = self.game.p2_pos
        board[p1_r][p1_c] = "1"
        board[p2_r][p2_c] = "2"
        
        # Build string
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(9)))
        for r, row in enumerate(board):
            lines.append(f"{r} " + " ".join(row))
        
        lines.append(f"\nP1 walls: {self.game.p1_walls_left}, P2 walls: {self.game.p2_walls_left}")
        lines.append(f"Current player: {self.game.current_player}")
        
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output
    
    def close(self):
        """Clean up resources."""
        pass


class SelfPlayEnv(QuoridorEnv):
    """
    Self-play environment where the agent plays against a copy of itself.
    
    Each step alternates between players. The observation is from the
    perspective of the current player (board is NOT flipped for simplicity).
    """
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 200):
        super().__init__(opponent=None, render_mode=render_mode, max_steps=max_steps)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step - returns observation for the next player."""
        self.steps += 1
        
        obs, reward, done, info = self.game.step(action)
        
        truncated = self.steps >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
