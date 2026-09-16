"""
Quoridor Game Logic Module.
Implements core game mechanics, rules enforcement, move validation, and board state transitions.
"""

import numpy as np
from typing import Optional
from core.state import QuoridorStateBuilder
from core.numba_utils import compute_legal_moves_mask


def flip_observation(obs: np.ndarray) -> np.ndarray:
    """
    Flip the observation vertically to normalize perspective.
    Current player always sees themselves as Player 1 moving from row 0 to 8.
    """
    flipped_obs = np.zeros_like(obs)

    # 1. Swap and flip player positions (Channels 0 and 1)
    flipped_obs[:, :, 0] = np.flip(obs[:, :, 1], axis=0)
    flipped_obs[:, :, 1] = np.flip(obs[:, :, 0], axis=0)

    # 2. Flip and shift walls (Channels 2 and 3)
    flipped_obs[:, :, 2] = np.roll(np.flip(obs[:, :, 2], axis=0), -1, axis=0)
    flipped_obs[:, :, 3] = np.roll(np.flip(obs[:, :, 3], axis=0), -1, axis=0)

    # 3. Swap and flip distance heatmaps (Channels 4 and 5)
    flipped_obs[:, :, 4] = np.flip(obs[:, :, 5], axis=0)
    flipped_obs[:, :, 5] = np.flip(obs[:, :, 4], axis=0)

    return flipped_obs


def flip_action(action: int) -> int:
    """Flip an action vertically to match flipped observation."""
    if action < 4:  # Pawn steps
        return {0: 2, 2: 0, 1: 1, 3: 3}[action]
    if action < 8:  # Straight jumps
        return {4: 6, 6: 4, 5: 5, 7: 7}[action]
    if action < 12:  # Diagonal jumps
        return {8: 10, 10: 8, 9: 11, 11: 9}[action]
    if action < 76:  # H-walls
        idx = action - 12
        r, c = idx // 8, idx % 8
        return 12 + (7 - r) * 8 + c
    if action < 140:  # V-walls
        idx = action - 76
        r, c = idx // 8, idx % 8
        return 76 + (7 - r) * 8 + c
    return action


def flip_mask(mask: np.ndarray) -> np.ndarray:
    """Flip the entire 140-dim action mask."""
    flipped_mask = np.zeros_like(mask)
    for i in range(140):
        if mask[i] > 0:
            flipped_mask[flip_action(i)] = mask[i]
    return flipped_mask


class QuoridorGame:
    """
    Core Quoridor game engine.
    
    Action Space: Discrete(140)
        - 0-3: Step N, E, S, W
        - 4-7: Jump N, E, S, W (straight jump over opponent)
        - 8-11: Slide NE, NW, SE, SW (diagonal moves)
        - 12-75: Horizontal Walls (64 positions)
        - 76-139: Vertical Walls (64 positions)
    """

    def __init__(self, walls_per_player: int = 10):
        self.initial_walls = walls_per_player
        self.state_builder = QuoridorStateBuilder() 
        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Resets the internal game state.
        
        Args:
            seed: Optional random seed for numpy RNG to guarantee reproducibility.
            
        Returns:
            Initial observation tensor (9x9x6).
        """
        if seed is not None:
            np.random.seed(seed)

        self.p1_pos = (0, 4)  # Row 0, Col 4
        self.p2_pos = (8, 4)  # Row 8, Col 4
        self.p1_walls_left = self.initial_walls
        self.p2_walls_left = self.initial_walls
        self.walls_h = []
        self.walls_v = []
        
        self.h_walls_mask = np.zeros((9, 9), dtype=np.int8)
        self.v_walls_mask = np.zeros((9, 9), dtype=np.int8)
        
        self.current_player = 1 
        self.done = False
        
        state = self._get_observation()
        self.p1_dist_prev = self._get_path_len(self.p1_pos, state[:, :, 4])
        self.p2_dist_prev = self._get_path_len(self.p2_pos, state[:, :, 5])
        
        return state

    def clone(self) -> "QuoridorGame":
        """
        Create a deep copy of internal game state without re-running Numba JIT.
        Fast state snapshot mechanism for tree search solvers (MCTS / Minimax).
        """
        new_game = QuoridorGame.__new__(QuoridorGame)
        new_game.initial_walls = self.initial_walls
        new_game.state_builder = self.state_builder
        new_game.p1_pos = self.p1_pos
        new_game.p2_pos = self.p2_pos
        new_game.p1_walls_left = self.p1_walls_left
        new_game.p2_walls_left = self.p2_walls_left
        new_game.walls_h = list(self.walls_h)
        new_game.walls_v = list(self.walls_v)
        new_game.h_walls_mask = self.h_walls_mask.copy()
        new_game.v_walls_mask = self.v_walls_mask.copy()
        new_game.current_player = self.current_player
        new_game.done = self.done
        new_game.p1_dist_prev = self.p1_dist_prev
        new_game.p2_dist_prev = self.p2_dist_prev
        return new_game

    def _get_observation(self) -> np.ndarray:
        """Helper to build the 9x9x6 Tensor from internal variables."""
        return self.state_builder.build_state(
            self.p1_pos, 
            self.p2_pos, 
            self.walls_h, 
            self.walls_v,
            self.h_walls_mask,
            self.v_walls_mask
        )
    
    def step(self, action: int):
        """
        Executes a move.
        Returns: observation, reward, done, info
        """
        if self.done:
            raise ValueError("Game is over. Call reset()!")

        # 1. Parse and Execute Action
        valid_move = self._apply_move(action)
        
        if not valid_move:
             obs = self._get_observation()
             return obs, -1.0, self.done, {"error": "Invalid Move"}

        # 2. Update distances for Reward Calculation
        state = self._get_observation()
        p1_dist_curr = self._get_path_len(self.p1_pos, state[:, :, 4])
        p2_dist_curr = self._get_path_len(self.p2_pos, state[:, :, 5])

        # 3. Calculate Reward
        reward = self._calculate_reward(valid_move, p1_dist_curr, p2_dist_curr)

        self.p1_dist_prev = p1_dist_curr
        self.p2_dist_prev = p2_dist_curr

        # 4. Check Game Over
        if self._check_win():
            self.done = True
            
        # 5. Switch Player (if game not over)
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        return state, reward, self.done, {}

    def _apply_move(self, action: int) -> bool:
        """Updates internal state based on action index. Returns True if valid."""
        if 0 <= action <= 11:
            target_pos = self._decode_pawn_move(action)
            if target_pos is None:
                return False
            
            if self.current_player == 1:
                self.p1_pos = target_pos
            else:
                self.p2_pos = target_pos
            return True

        w_action = action - 12
        if self.current_player == 1 and self.p1_walls_left <= 0:
            return False
        if self.current_player == 2 and self.p2_walls_left <= 0:
            return False

        wall_type = 'h' if w_action < 64 else 'v'
        idx = w_action if w_action < 64 else w_action - 64
        
        r = idx // 8
        c = idx % 8
        
        if wall_type == 'h':
            self.walls_h.append((r, c))
            self.h_walls_mask[r, c] = 1
        else:
            self.walls_v.append((r, c))
            self.v_walls_mask[r, c] = 1
            
        if self.current_player == 1:
            self.p1_walls_left -= 1
        else:
            self.p2_walls_left -= 1
            
        return True
    
    def _decode_pawn_move(self, action: int):
        """Translate action (0-11) to a target coordinate IF valid."""
        curr = self.p1_pos if self.current_player == 1 else self.p2_pos
        opp = self.p2_pos if self.current_player == 1 else self.p1_pos
        
        valid_moves = self._get_valid_pawn_moves(curr, opp)
        
        deltas = {
            0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1),
            4: (-2, 0), 5: (0, 2), 6: (2, 0), 7: (0, -2),
            8: (-1, 1), 9: (-1, -1), 10: (1, 1), 11: (1, -1)
        }
        
        if action not in deltas:
            return None
            
        dr, dc = deltas[action]
        desired_pos = (curr[0] + dr, curr[1] + dc)
        
        if desired_pos in valid_moves:
            return desired_pos
        return None

    def _get_valid_pawn_moves(self, curr, opp):
        """Returns set of valid target (r, c) positions for 'curr' pawn."""
        moves = set()
        neighbors = self.state_builder._get_valid_neighbors(curr[0], curr[1], self.h_walls_mask, self.v_walls_mask)
        
        for (nr, nc) in neighbors:
            if (nr, nc) == opp:
                opp_neighbors = self.state_builder._get_valid_neighbors(nr, nc, self.h_walls_mask, self.v_walls_mask)
                dr, dc = nr - curr[0], nc - curr[1]
                jump_dest = (nr + dr, nc + dc)
                
                if jump_dest in opp_neighbors:
                    moves.add(jump_dest)
                else:
                    for (onr, onc) in opp_neighbors:
                         if (onr, onc) != curr: 
                             moves.add((onr, onc))
            else:
                moves.add((nr, nc))
                
        return moves

    def _get_path_len(self, pos, heatmap) -> int:
        r, c = pos
        dist_norm = heatmap[r, c]
        dist = int(round(dist_norm * self.state_builder.max_dist))
        if dist >= int(self.state_builder.max_dist):
            return 999
        return dist

    def _calculate_reward(self, valid_move, p1_dist_curr, p2_dist_curr):
        if not valid_move:
            return -0.5

        p1_diff = self.p1_dist_prev - p1_dist_curr
        p2_diff = self.p2_dist_prev - p2_dist_curr

        if self.current_player == 1:
            my_progress = p1_diff
            opp_setback = -p2_diff
        else:
            my_progress = p2_diff
            opp_setback = -p1_diff

        w_progress = 0.1
        w_setback = 0.05
        
        shaping = (w_progress * my_progress) + (w_setback * opp_setback)
        step_cost = -0.01 

        reward = step_cost + shaping
        
        if self._check_win():
            reward += 10.0
        
        return reward

    def _check_win(self) -> bool:
        if self.p1_pos[0] == 8:
            return True
        if self.p2_pos[0] == 0:
            return True
        return False
        
    def get_legal_moves(self) -> np.ndarray:
        """
        Returns a binary mask [140] of valid actions.
        Uses Numba optimized logic for speed.
        """
        walls_left = self.p1_walls_left if self.current_player == 1 else self.p2_walls_left
        
        return compute_legal_moves_mask(
            self.p1_pos[0], self.p1_pos[1],
            self.p2_pos[0], self.p2_pos[1],
            self.current_player,
            walls_left,
            self.h_walls_mask,
            self.v_walls_mask
        )
