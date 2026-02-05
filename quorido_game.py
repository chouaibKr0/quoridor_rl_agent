"""
Quorido Game Logic Module
Implements the core game mechanics and board state management
"""

import numpy as np
import collections

class QuoridorStateBuilder:
    # TODO: Review carefully this class
    def __init__(self):
        # 9x9 board
        self.height = 9
        self.width = 9
        # Max path length for normalization (approx 81 squares)
        self.max_dist = 81.0 

    def get_shortest_path_heatmap(self, start_pos: tuple[int, int], goal_row: int, walls_h: list[tuple[int, int]], walls_v: list[tuple[int, int]]):
        """
        Runs BFS to calculate distance from EVERY square to the GOAL.
        Returns a 9x9 grid where value = distance to goal.
        """
        distances = np.full((self.height, self.width), self.max_dist)
        queue = collections.deque()
        
        # Initialize BFS from the GOAL line (backward search)
        # For P1 (starts top), goal is row 8 (bottom)
        # For P2 (starts bottom), goal is row 0 (top)
        
        # If goal is specific row:
        for col in range(9):
            distances[goal_row, col] = 0
            queue.append((goal_row, col))
            
        while queue:
            r, c = queue.popleft()
            dist = distances[r, c]
            
            # Check neighbors (Up, Down, Left, Right)
            # You must add logic here to check if a WALL blocks the move
            neighbors = self._get_valid_neighbors(r, c, walls_h, walls_v)
            
            for nr, nc in neighbors:
                if distances[nr, nc] == self.max_dist: # Not visited
                    distances[nr, nc] = dist + 1
                    queue.append((nr, nc))
        #TODO          
        # Invert/Normalize: Close to goal = 1.0, Far = 0.0
        # Or just Raw Normalized Distance: Close = 0.0, Far = 1.0
        # Let's use Normalized Distance (0 means at goal)
        return distances / self.max_dist

    def _get_valid_neighbors(self, r:int, c:int, walls_h: list[tuple[int, int]], walls_v: list[tuple[int, int]]):
        # Placeholder for your wall collision logic
        # Returns list of (r, c) tuples
        moves = []
        #TODO ... logic to check walls_h and walls_v ...
        return moves

    def build_state(self, p1_pos:tuple[int, int], p2_pos:tuple[int, int], walls_h: list[tuple[int, int]], walls_v: list[tuple[int, int]]):
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
        # walls_h are typically (r, c) of the top-left corner of the wall
        for (r, c) in walls_h:
            if 0 <= r < 9 and 0 <= c < 9:
                state[r, c, 2] = 1.0

        # --- Channel 3: Vertical Walls ---
        for (r, c) in walls_v:
            if 0 <= r < 9 and 0 <= c < 9:
                state[r, c, 3] = 1.0

        # --- Channel 4: P1 Distance Map (Heatmap) ---
        # P1 wants to get to row 8
        dist_map_p1 = self.get_shortest_path_heatmap(p1_pos, 8, walls_h, walls_v)
        state[:, :, 4] = dist_map_p1

        # --- Channel 5: P2 Distance Map (Heatmap) ---
        # P2 wants to get to row 0
        dist_map_p2 = self.get_shortest_path_heatmap(p2_pos, 0, walls_h, walls_v)
        state[:, :, 5] = dist_map_p2

        return state
    def get_shortest_path_length(self, start_pos: tuple[int, int], goal_row: int, walls_h: list[tuple[int, int]], walls_v: list[tuple[int, int]]) -> int:
        #TODO: use heatmap to get shortest path length from start_pos to goal_row
        ...
    


class QuoridorGame:
    # TODO: Review and recheck the logic, especially the reward shaping part.
    def __init__(self, walls_per_player=10):
        self.initial_walls = walls_per_player
        self.state_builder = QuoridorStateBuilder() # The helper class we defined before
        self.reset()
    def get_path_len(self, pos, goal_row) -> int:
        """Returns the shortest path length (int) from pos to goal_row."""
        # Use your BFS logic here. 
        # Return 999 if no path (though technically that's illegal in Quoridor)
        return self.state_builder.get_shortest_path_length(pos, goal_row, self.walls_h, self.walls_v)
    def reset(self):
        """
        Resets the INTERNAL game state. 
        Returns the initial OBSERVATION (the tensor).
        """
        # 1. Internal Logic Variables (Lightweight)
        self.p1_pos = (0, 4)  # Row 0, Col 4
        self.p2_pos = (8, 4)  # Row 8, Col 4
        self.p1_walls_left = self.initial_walls
        self.p2_walls_left = self.initial_walls
        self.walls_h = []  # Set of (r, c) tuples
        self.walls_v = []  # Set of (r, c) tuples
        self.current_player = 1 
        self.done = False
        self.p1_dist_prev = self.get_path_len(self.p1_pos, 8) 
        self.p2_dist_prev = self.get_path_len(self.p2_pos, 0)
        # 2. Return the first observation for the agent
        return self._get_observation()

    def _get_observation(self):
        """
        Helper to build the 9x9x6 Tensor from internal variables.
        This is called by reset() and step().
        """
        return self.state_builder.build_state(
            self.p1_pos, 
            self.p2_pos, 
            self.walls_h, 
            self.walls_v
        )

    def step(self, action):
        """
        Executes a move.
        Returns: observation, reward, done, info
        """
        if self.done:
            raise ValueError("Game is over. Call reset()!")

        # 1. Parse and Execute Action (Update internal vars)
        # (We will implement the logic inside _apply_move later)
        valid_move = self._apply_move(action)

        # 2. Calculate Reward

        # A. Calculate NEW distances (after the move)
        p1_dist_curr = self.get_path_len(self.p1_pos, 8)
        p2_dist_curr = self.get_path_len(self.p2_pos, 0)
        
        # B. Calculate the Difference (The "Delta")
        # Did P1 get closer? (Old - New) -> Positive is good
        p1_progress = self.p1_dist_prev - p1_dist_curr
        p1_damage = p1_dist_curr - self.p1_dist_prev
        # Did P2 get pushed back? (New - Old) -> Positive is good
        p2_progress = self.p2_dist_prev - p2_dist_curr
        p2_damage = p2_dist_curr - self.p2_dist_prev
        
        # C. Your Formula
        # If I am Player 1:
        if self.current_player == 1:
            # Reward = (My Progress) + (Damage to Opponent)
            shaping = 0.05 * (p1_progress + p2_damage)
        else:
            # If I am Player 2, logic is flipped
            shaping = 0.05 * (p2_progress + p1_damage)

        # D. Add to standard rewards
        reward = -0.01 + shaping
        if self._check_win():
            reward += 1.0

        self.p1_dist_prev = p1_dist_curr
        self.p2_dist_prev = p2_dist_curr



        # 3. Check Game Over
        if self._check_win():
            self.done = True
            
        # 4. Switch Player (if game not over)
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        # 5. Return the standard RL tuple
        # Observation (Tensor), Reward (Float), Done (Bool), Info (Dict)
        return self._get_observation(), reward, self.done, {}

    def _apply_move(self, action):
        """Updates p1_pos, walls_h, etc. based on action index."""
        # Logic to decode 'action' integer to a game move
        # Update self.p1_pos or self.walls_h...
        return True # or False if move was somehow invalid (though we will mask those)

    def _calculate_reward(self, valid_move):
        # Your +1/-1 logic goes here
        if self._check_win():
            return 1.0
        return -0.01 # Step cost

    def _check_win(self):
        # P1 wins if row == 8, P2 wins if row == 0
        if self.current_player == 1 and self.p1_pos[0] == 8:
            return True
        if self.current_player == 2 and self.p2_pos[0] == 0:
            return True
        return False
        
    def get_legal_moves(self):
        """
        Returns a binary mask [1, 0, 0, 1...] of valid actions.
        Used for Action Masking.
        """
        mask = np.zeros(137) # Size of your action space
        # Fill mask with 1s for valid moves...
        return mask
