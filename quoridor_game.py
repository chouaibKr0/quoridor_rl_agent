"""
Quorido Game Logic Module
Implements the core game mechanics and board state management
"""

import numpy as np
import collections
from numba_utils import bfs_distances_numba, check_path_exists_numba, get_valid_neighbors_numba

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
    # H-wall at r (between r, r+1) becomes 7-r (between 8-r, 7-r)
    flipped_obs[:, :, 2] = np.roll(np.flip(obs[:, :, 2], axis=0), -1, axis=0)
    flipped_obs[:, :, 3] = np.roll(np.flip(obs[:, :, 3], axis=0), -1, axis=0)

    # 3. Swap and flip distance heatmaps (Channels 4 and 5)
    # P2 heatmap (dist to 0) becomes dist to 8 on flipped board
    flipped_obs[:, :, 4] = np.flip(obs[:, :, 5], axis=0)
    flipped_obs[:, :, 5] = np.flip(obs[:, :, 4], axis=0)

    return flipped_obs

def flip_action(action: int) -> int:
    """Flip an action vertically to match flipped observation."""
    if action < 4: # Pawn steps
        # 0:N, 1:E, 2:S, 3:W -> N and S swap
        return {0: 2, 2: 0, 1: 1, 3: 3}[action]
    if action < 8: # Straight jumps
        # 4:N, 5:E, 6:S, 7:W -> N and S swap
        return {4: 6, 6: 4, 5: 5, 7: 7}[action]
    if action < 12: # Diagonal jumps
        # 8:NE, 9:NW, 10:SE, 11:SW -> N/S swap
        return {8: 10, 10: 8, 9: 11, 11: 9}[action]
    if action < 76: # H-walls
        idx = action - 12
        r, c = idx // 8, idx % 8
        return 12 + (7 - r) * 8 + c
    if action < 140: # V-walls
        idx = action - 76
        r, c = idx // 8, idx % 8
        return 76 + (7 - r) * 8 + c
    return action

def flip_mask(mask: np.ndarray) -> np.ndarray:
    """Flip the entire 140-dim action mask."""
    flipped_mask = np.zeros_like(mask)
    # Use the mapping to flip each bit in the mask
    for i in range(140):
        if mask[i] > 0:
            flipped_mask[flip_action(i)] = mask[i]
    return flipped_mask

class QuoridorStateBuilder:
    # TODO: Review carefully this class
    def __init__(self):
        # 9x9 board
        self.height = 9
        self.width = 9
        # Max path length for normalization (approx 81 squares)
        self.max_dist = 81.0 

    def get_shortest_path_heatmap(self, start_pos: tuple[int, int], goal_row: int, h_walls_mask, v_walls_mask):
        """
        Runs BFS to calculate distance from EVERY square to the GOAL.
        Returns a 9x9 grid where value = distance to goal.
        """
        # Call Numba optimized BFS
        # Note: Numba function takes masks directly
        # p1_pos is not needed for heatmap generation, just goal row and walls
        
        # We need distances FROM goal TO everywhere.
        # Our Numba BFS initializes queue with goal_row and floods outwards.
        # So distances[r, c] is distance to goal.
        return bfs_distances_numba(start_pos[0], start_pos[1], goal_row, h_walls_mask, v_walls_mask, self.max_dist)

    def _get_valid_neighbors(self, r:int, c:int, h_walls_mask, v_walls_mask):
        # Wrapper for Numba optimized neighbor check
        nbs_arr, count = get_valid_neighbors_numba(r, c, h_walls_mask, v_walls_mask)
        moves = []
        for i in range(count):
            moves.append((nbs_arr[i, 0], nbs_arr[i, 1]))
        return moves

    def build_state(self, p1_pos:tuple[int, int], p2_pos:tuple[int, int], walls_h: list[tuple[int, int]], walls_v: list[tuple[int, int]], h_walls_mask, v_walls_mask):
        """
        Constructs the 9x9x6 observation tensor.
        p1_pos, p2_pos: tuples (row, col)
        walls_h, walls_v: lists or sets of wall coordinates
        """
        # Initialize 9x9x6 tensor with zeros
        # Shape: (Height, Width, Channels)
        state = np.zeros((9, 9, 6), dtype=np.float32)

        # --- Channel 0: Player 1 Position ---
        r1, c1 = p1_pos
        state[r1, c1, 0] = 1.0

        # --- Channel 1: Player 2 Position ---
        r2, c2 = p2_pos
        state[r2, c2, 1] = 1.0

        # --- Channel 2: Horizontal Walls ---
        # USE MASKS directly if aligned?
        # The mask is 9x9. walls_h list also maps to 9x9.
        state[:, :, 2] = h_walls_mask.astype(np.float32)

        # --- Channel 3: Vertical Walls ---
        state[:, :, 3] = v_walls_mask.astype(np.float32)

        # --- Channel 4: P1 Distance Map (Heatmap) ---
        # P1 wants to get to row 8
        dist_map_p1 = self.get_shortest_path_heatmap(p1_pos, 8, h_walls_mask, v_walls_mask)
        state[:, :, 4] = dist_map_p1

        # --- Channel 5: P2 Distance Map (Heatmap) ---
        # P2 wants to get to row 0
        dist_map_p2 = self.get_shortest_path_heatmap(p2_pos, 0, h_walls_mask, v_walls_mask)
        state[:, :, 5] = dist_map_p2

        return state


class QuoridorGame:
    # Action Space: 140
    # 0-3: Step N, E, S, W
    # 4-7: Jump N, E, S, W (Straight jump over opponent)
    # 8-11: Slide NE, NW, SE, SW (Diagonal move)
    # 12-75: Horizontal Walls (64 positions)
    # 76-139: Vertical Walls (64 positions)

    def __init__(self, walls_per_player=10):
        self.initial_walls = walls_per_player
        self.state_builder = QuoridorStateBuilder() 
        self.reset()

    def reset(self):
        """
        Resets the INTERNAL game state. 
        Returns the initial OBSERVATION (the tensor).
        """
        self.p1_pos = (0, 4)  # Row 0, Col 4
        self.p2_pos = (8, 4)  # Row 8, Col 4
        self.p1_walls_left = self.initial_walls
        self.p2_walls_left = self.initial_walls
        self.walls_h = []  # List of (r, c)
        self.walls_v = []  # List of (r, c)
        
        # --- OPTIMIZATION: Maintain masks ---
        self.h_walls_mask = np.zeros((9, 9), dtype=np.int8)
        self.v_walls_mask = np.zeros((9, 9), dtype=np.int8)
        
        self.current_player = 1 
        self.done = False
        
        # Calculate initial distances
        # Note: We use existing methods but need current state
        state = self._get_observation()
        # Channel 4 is P1 dist map, Channel 5 is P2 dist map
        # Dist map is normalized. 
        self.p1_dist_prev = self._get_path_len(self.p1_pos, state[:,:,4])
        self.p2_dist_prev = self._get_path_len(self.p2_pos, state[:,:,5])
        
        return state

    def _get_observation(self):
        """
        Helper to build the 9x9x6 Tensor from internal variables.
        """
        return self.state_builder.build_state(
            self.p1_pos, 
            self.p2_pos, 
            self.walls_h, 
            self.walls_v,
            self.h_walls_mask,
            self.v_walls_mask
        )
    
    def step(self, action):
        """
        Executes a move.
        Returns: observation, reward, done, info
        """
        if self.done:
            raise ValueError("Game is over. Call reset()!")

        # 1. Parse and Execute Action
        valid_move = self._apply_move(action)
        
        if not valid_move:
             # This should ideally be blocked by action masking, but as a fallback:
             # Use a heavy penalty and standard return
             obs = self._get_observation()
             return obs, -1.0, self.done, {"error": "Invalid Move"}

        # 2. Update distances for Reward Calculation
        state = self._get_observation()
        # Note: state builder returns normalized distances
        p1_dist_curr = self._get_path_len(self.p1_pos, state[:, :, 4])
        p2_dist_curr = self._get_path_len(self.p2_pos, state[:, :, 5])

        # 3. Calculate Reward
        reward = self._calculate_reward(valid_move, p1_dist_curr, p2_dist_curr)

        self.p1_dist_prev = p1_dist_curr
        self.p2_dist_prev = p2_dist_curr

        # 4. Check Game Over
        if self._check_win():
            self.done = True
            # Bonus for winning is added in calculate_reward
            
        # 5. Switch Player (if game not over)
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        return state, reward, self.done, {}

    def _apply_move(self, action):
        """Updates internal state based on action index. Returns True if valid."""
        # Action Decoding
        # 0-11: Pawn Moves
        if 0 <= action <= 11:
            target_pos = self._decode_pawn_move(action)
            if target_pos is None:
                return False
            
            # Update position
            if self.current_player == 1:
                self.p1_pos = target_pos
            else:
                self.p2_pos = target_pos
            return True

        # 12-139: Wall Placements
        # Offset by 12
        w_action = action - 12
        if self.current_player == 1 and self.p1_walls_left <= 0:
            return False
        if self.current_player == 2 and self.p2_walls_left <= 0:
            return False

        wall_type = 'h' if w_action < 64 else 'v'
        idx = w_action if w_action < 64 else w_action - 64
        
        # Decode (r, c) from idx (0..63)
        # r in 0..7, c in 0..7
        r = idx // 8
        c = idx % 8
        
        # Add Wall
        if wall_type == 'h':
            self.walls_h.append((r, c))
            self.h_walls_mask[r, c] = 1
        else:
            self.walls_v.append((r, c))
            self.v_walls_mask[r, c] = 1
            
        # Decrement walls
        if self.current_player == 1:
            self.p1_walls_left -= 1
        else:
            self.p2_walls_left -= 1
            
        return True
    
    def _decode_pawn_move(self, action):
        """
        Translate action (0-11) to a target coordinate IF valid.
        """
        # Current and Opponent positions
        curr = self.p1_pos if self.current_player == 1 else self.p2_pos
        opp = self.p2_pos if self.current_player == 1 else self.p1_pos
        
        valid_moves = self._get_valid_pawn_moves(curr, opp)
        
        # Mapping Actions to Delta (dr, dc)
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
        """
        Returns set of valid target (r, c) positions for 'curr' pawn.
        """
        moves = set()
        
        # 1. Get graph neighbors (blocked by walls?)
        neighbors = self.state_builder._get_valid_neighbors(curr[0], curr[1], self.h_walls_mask, self.v_walls_mask)
        
        for (nr, nc) in neighbors:
            if (nr, nc) == opp:
                # 2. Opponent is here. Try Jump/Slide.
                opp_neighbors = self.state_builder._get_valid_neighbors(nr, nc, self.h_walls_mask, self.v_walls_mask)
                
                # Direction of jump: (nr-cr, nc-cc)
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
            return -0.5  # Increase penalty for invalid moves to force learning valid rules

        # Calculate raw changes (positive = good for that player)
        # Distance decreased = Progress
        p1_diff = self.p1_dist_prev - p1_dist_curr
        p2_diff = self.p2_dist_prev - p2_dist_curr

        if self.current_player == 1:
            my_progress = p1_diff
            opp_setback = -p2_diff  # Negative p2_diff means p2 distance increased (good for p1)
        else:
            my_progress = p2_diff
            opp_setback = -p1_diff

        # Weights - Reduced to ensure win/loss is the primary signal
        w_progress = 0.1
        w_setback = 0.05
        
        # Shaping
        shaping = (w_progress * my_progress) + (w_setback * opp_setback)

        # Step cost
        step_cost = -0.01 

        reward = step_cost + shaping
        
        # Huge bonus for winning
        # Note: Loss penalty is handled in the environment wrapper
        if self._check_win():
            reward += 10.0
        
        return reward



    def _check_win(self):
        if self.p1_pos[0] == 8:
            return True
        if self.p2_pos[0] == 0:
            return True
        return False
        
    def get_legal_moves(self):
        """
        Returns a binary mask [140] of valid actions.
        Uses Numba optimized logic for speed.
        """
        walls_left = self.p1_walls_left if self.current_player == 1 else self.p2_walls_left
        
        from numba_utils import compute_legal_moves_mask

        return compute_legal_moves_mask(
            self.p1_pos[0], self.p1_pos[1],
            self.p2_pos[0], self.p2_pos[1],
            self.current_player,
            walls_left,
            self.h_walls_mask,
            self.v_walls_mask
        )
    
    def _has_path(self, start_pos, goal_row, walls_h, walls_v):
        """
        DEPRECATED: Use check_path_exists_numba instead.
        """
        pass
