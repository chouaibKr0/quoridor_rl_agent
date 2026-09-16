"""
Quoridor State Construction Module.
Builds the 9x9x6 state observation tensor representing the current game board.
"""

import numpy as np
from core.numba_utils import bfs_distances_numba, get_valid_neighbors_numba


class QuoridorStateBuilder:
    """
    Constructs the 9x9x6 observation tensor for the Quoridor environment.
    
    Channels:
        - 0: Player 1 position (one-hot)
        - 1: Player 2 position (one-hot)
        - 2: Horizontal walls mask
        - 3: Vertical walls mask
        - 4: Player 1 distance heatmap (BFS distance to row 8)
        - 5: Player 2 distance heatmap (BFS distance to row 0)
    """
    def __init__(self):
        self.height = 9
        self.width = 9
        self.max_dist = 81.0 

    def get_shortest_path_heatmap(self, start_pos: tuple[int, int], goal_row: int, h_walls_mask, v_walls_mask):
        """
        Runs BFS to calculate distance from EVERY square to the GOAL.
        Returns a 9x9 grid where value = distance to goal.
        """
        return bfs_distances_numba(start_pos[0], start_pos[1], goal_row, h_walls_mask, v_walls_mask, self.max_dist)

    def _get_valid_neighbors(self, r: int, c: int, h_walls_mask, v_walls_mask):
        """Wrapper for Numba optimized neighbor check."""
        nbs_arr, count = get_valid_neighbors_numba(r, c, h_walls_mask, v_walls_mask)
        moves = []
        for i in range(count):
            moves.append((nbs_arr[i, 0], nbs_arr[i, 1]))
        return moves

    def build_state(self, p1_pos: tuple[int, int], p2_pos: tuple[int, int], walls_h: list[tuple[int, int]], walls_v: list[tuple[int, int]], h_walls_mask, v_walls_mask):
        """
        Constructs the 9x9x6 observation tensor.
        
        Args:
            p1_pos: (row, col) tuple for Player 1
            p2_pos: (row, col) tuple for Player 2
            walls_h: List of horizontal wall coordinates
            walls_v: List of vertical wall coordinates
            h_walls_mask: 9x9 numpy array indicating horizontal wall presence
            v_walls_mask: 9x9 numpy array indicating vertical wall presence
            
        Returns:
            np.ndarray of shape (9, 9, 6) and dtype float32
        """
        state = np.zeros((9, 9, 6), dtype=np.float32)

        # --- Channel 0: Player 1 Position ---
        r1, c1 = p1_pos
        state[r1, c1, 0] = 1.0

        # --- Channel 1: Player 2 Position ---
        r2, c2 = p2_pos
        state[r2, c2, 1] = 1.0

        # --- Channel 2: Horizontal Walls ---
        state[:, :, 2] = h_walls_mask.astype(np.float32)

        # --- Channel 3: Vertical Walls ---
        state[:, :, 3] = v_walls_mask.astype(np.float32)

        # --- Channel 4: P1 Distance Map (Heatmap) ---
        dist_map_p1 = self.get_shortest_path_heatmap(p1_pos, 8, h_walls_mask, v_walls_mask)
        state[:, :, 4] = dist_map_p1

        # --- Channel 5: P2 Distance Map (Heatmap) ---
        dist_map_p2 = self.get_shortest_path_heatmap(p2_pos, 0, h_walls_mask, v_walls_mask)
        state[:, :, 5] = dist_map_p2

        return state
