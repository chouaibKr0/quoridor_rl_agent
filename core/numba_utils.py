"""
Numba-accelerated helper utilities for Quoridor pathfinding and move mask computation.
"""

import numpy as np
from numba import njit

@njit
def get_valid_neighbors_numba(r, c, h_walls_mask, v_walls_mask):
    """
    Returns a fixed-size array of neighbors and a count, to avoid list allocation.
    Max neighbors is 4.
    Returns: (neighbors_array, count)
    neighbors_array is shaped (4, 2)
    """
    neighbors = np.zeros((4, 2), dtype=np.int32)
    count = 0
    
    # Up: (r-1, c)
    if r > 0:
        blocked = False
        if h_walls_mask[r-1, c] == 1: blocked = True
        if c > 0 and h_walls_mask[r-1, c-1] == 1: blocked = True
        
        if not blocked:
            neighbors[count, 0] = r - 1
            neighbors[count, 1] = c
            count += 1

    # Down: (r+1, c)
    if r < 8:
        blocked = False
        if h_walls_mask[r, c] == 1: blocked = True
        if c > 0 and h_walls_mask[r, c-1] == 1: blocked = True
        
        if not blocked:
            neighbors[count, 0] = r + 1
            neighbors[count, 1] = c
            count += 1

    # Left: (r, c-1)
    if c > 0:
        blocked = False
        if v_walls_mask[r, c-1] == 1: blocked = True
        if r > 0 and v_walls_mask[r-1, c-1] == 1: blocked = True
        
        if not blocked:
            neighbors[count, 0] = r
            neighbors[count, 1] = c - 1
            count += 1

    # Right: (r, c+1)
    if c < 8:
        blocked = False
        if v_walls_mask[r, c] == 1: blocked = True
        if r > 0 and v_walls_mask[r-1, c] == 1: blocked = True
        
        if not blocked:
            neighbors[count, 0] = r
            neighbors[count, 1] = c + 1
            count += 1
            
    return neighbors, count

@njit
def bfs_distances_numba(start_row, start_col, goal_row, h_walls_mask, v_walls_mask, max_dist=81.0):
    """
    Computes BFS distances from EVERY cell to goal_row (heatmap generation).
    """
    distances = np.full((9, 9), max_dist, dtype=np.float32)
    q_arr = np.zeros((100, 2), dtype=np.int32)
    head = 0
    tail = 0
    
    # Initialize with goal row
    for c in range(9):
        distances[goal_row, c] = 0.0
        q_arr[tail, 0] = goal_row
        q_arr[tail, 1] = c
        tail += 1
        
    while head < tail:
        r = q_arr[head, 0]
        c = q_arr[head, 1]
        head += 1
        
        dist = distances[r, c]
        
        # Get neighbors
        nbs, count = get_valid_neighbors_numba(r, c, h_walls_mask, v_walls_mask)
        
        for i in range(count):
            nr = nbs[i, 0]
            nc = nbs[i, 1]
            if distances[nr, nc] == max_dist:
                distances[nr, nc] = dist + 1.0
                q_arr[tail, 0] = nr
                q_arr[tail, 1] = nc
                tail += 1
                
    return distances / max_dist

@njit
def check_path_exists_numba(start_r, start_c, goal_row, h_walls_mask, v_walls_mask):
    """
    Quick BFS to check if start_pos can reach goal_row.
    Returns True/False.
    """
    visited = np.zeros((9, 9), dtype=np.int8)
    q_arr = np.zeros((100, 2), dtype=np.int32)
    head = 0
    tail = 0
    
    q_arr[tail, 0] = start_r
    q_arr[tail, 1] = start_c
    tail += 1
    visited[start_r, start_c] = 1
    
    while head < tail:
        r = q_arr[head, 0]
        c = q_arr[head, 1]
        head += 1
        
        if r == goal_row:
            return True
            
        nbs, count = get_valid_neighbors_numba(r, c, h_walls_mask, v_walls_mask)
        for i in range(count):
            nr = nbs[i, 0]
            nc = nbs[i, 1]
            if visited[nr, nc] == 0:
                visited[nr, nc] = 1
                q_arr[tail, 0] = nr
                q_arr[tail, 1] = nc
                tail += 1
                
    return False

@njit
def get_valid_pawn_moves_numba(curr_r, curr_c, opp_r, opp_c, h_walls_mask, v_walls_mask):
    """
    Returns valid moves list for the pawn.
    Returns: (moves_array, count)
    """
    valid_moves = np.zeros((10, 2), dtype=np.int32)
    vm_count = 0
    
    # 1. Get graph neighbors
    nbs, count = get_valid_neighbors_numba(curr_r, curr_c, h_walls_mask, v_walls_mask)
    
    for i in range(count):
        nr = nbs[i, 0]
        nc = nbs[i, 1]
        
        if nr == opp_r and nc == opp_c:
            # 2. Opponent is here. Try Jump/Slide.
            opp_nbs, opp_count = get_valid_neighbors_numba(nr, nc, h_walls_mask, v_walls_mask)
            
            # Jump direction
            dr = nr - curr_r
            dc = nc - curr_c
            jump_r = nr + dr
            jump_c = nc + dc
            
            # Check if jump destination is in opponent's neighbors
            can_jump = False
            for k in range(opp_count):
                if opp_nbs[k, 0] == jump_r and opp_nbs[k, 1] == jump_c:
                    can_jump = True
                    break
            
            if can_jump:
                valid_moves[vm_count, 0] = jump_r
                valid_moves[vm_count, 1] = jump_c
                vm_count += 1
            else:
                # Add all opponent neighbors EXCEPT ourselves
                for k in range(opp_count):
                    onr = opp_nbs[k, 0]
                    onc = opp_nbs[k, 1]
                    if not (onr == curr_r and onc == curr_c):
                        valid_moves[vm_count, 0] = onr
                        valid_moves[vm_count, 1] = onc
                        vm_count += 1
        else:
            valid_moves[vm_count, 0] = nr
            valid_moves[vm_count, 1] = nc
            vm_count += 1
            
    return valid_moves, vm_count


@njit
def compute_legal_moves_mask(
    p1_r, p1_c, p2_r, p2_c, 
    current_player, 
    walls_left, 
    h_walls_mask, v_walls_mask
):
    """
    Generates the entire 140-dim action mask in one go.
    """
    mask = np.zeros(140, dtype=np.float32)
    
    # --- 1. Pawn Moves (0-11) ---
    curr_r, curr_c = (p1_r, p1_c) if current_player == 1 else (p2_r, p2_c)
    opp_r, opp_c   = (p2_r, p2_c) if current_player == 1 else (p1_r, p1_c)
    
    valid_moves, vm_count = get_valid_pawn_moves_numba(curr_r, curr_c, opp_r, opp_c, h_walls_mask, v_walls_mask)
    
    dr = np.array([-1, 0, 1, 0, -2, 0, 2, 0], dtype=np.int32)
    dc = np.array([0, 1, 0, -1, 0, 2, 0, -2], dtype=np.int32)
    
    # Check actions 0-7 (Pawn steps and straight jumps)
    for a in range(8):
        tr = curr_r + dr[a]
        tc = curr_c + dc[a]
        
        is_valid = False
        for k in range(vm_count):
            if valid_moves[k, 0] == tr and valid_moves[k, 1] == tc:
                is_valid = True
                break
        if is_valid:
            mask[a] = 1.0

    # --- 2. Wall Moves (12-139) ---
    if walls_left > 0:
        for w_idx in range(128):
            wall_type_h = True if w_idx < 64 else False
            local_idx = w_idx if w_idx < 64 else w_idx - 64
            r = local_idx // 8
            c = local_idx % 8
            
            # A. Overlap Check
            is_valid_pos = True
            if wall_type_h:
                if h_walls_mask[r, c] == 1: is_valid_pos = False
                if c > 0 and h_walls_mask[r, c-1] == 1: is_valid_pos = False
                if c < 8 and h_walls_mask[r, c+1] == 1: is_valid_pos = False
                if v_walls_mask[r, c] == 1: is_valid_pos = False
            else: # Vertical
                if v_walls_mask[r, c] == 1: is_valid_pos = False
                if r > 0 and v_walls_mask[r-1, c] == 1: is_valid_pos = False
                if r < 8 and v_walls_mask[r+1, c] == 1: is_valid_pos = False
                if h_walls_mask[r, c] == 1: is_valid_pos = False
            
            if not is_valid_pos:
                continue
                
            # B. Pathblocking Check
            if wall_type_h:
                h_walls_mask[r, c] = 1
            else:
                v_walls_mask[r, c] = 1
                
            has_p1 = check_path_exists_numba(p1_r, p1_c, 8, h_walls_mask, v_walls_mask)
            
            if has_p1:
                 has_p2 = check_path_exists_numba(p2_r, p2_c, 0, h_walls_mask, v_walls_mask)
                 if has_p2:
                     mask[12 + w_idx] = 1.0
            
            # Revert wall
            if wall_type_h:
                h_walls_mask[r, c] = 0
            else:
                v_walls_mask[r, c] = 0

    return mask
