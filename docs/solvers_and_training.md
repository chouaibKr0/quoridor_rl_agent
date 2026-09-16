# Solvers and Training

This document details the AI agents (solvers) available in the Quoridor framework and the Reinforcement Learning (RL) training pipeline.

## 1. Solver Registry

The system employs a factory pattern to instantiate agents dynamically based on string names and configuration dictionaries. This allows the UI, training loop, and experiment runner to request agents identically.

```python
from solvers.registry import get_solver
agent = get_solver("strategic", {"wall_prob": 0.7})
```

All solvers inherit from `solvers.base.BaseAgent` and must implement:
- `select_action(observation, action_mask)`
- `get_config()` (for telemetry logging)

## 2. Heuristic Solvers

Located in `solvers/heuristic/`, these agents use hand-crafted rules without deep search.

1. **RandomAgent** (`random`): Samples uniformly from the legal `action_mask`.
2. **DijkstraAgent** (`dijkstra`): Greedily takes the pawn move that minimizes its path length to the goal, leveraging the environment's pre-computed distance heatmaps. Does not place walls.
3. **StrategicAgent** (`strategic`): Attempts to place a wall that increases the opponent's path length. If no such wall exists (or based on its `wall_prob` config), it falls back to Dijkstra movement.

## 3. Tree Search Solvers

Located in `solvers/search/`, these agents simulate future states using the Numba-accelerated core game logic. They expose a `last_move_stats` property to log compute efficiency (e.g. nodes expanded).

1. **MinimaxAgent** (`minimax`): Explores the game tree to a fixed depth. Evaluates leaf nodes using the heuristic: `score = my_path_len - opponent_path_len`. Restricts wall branching to the top-$K$ most disruptive walls to prevent explosion.
2. **AlphaBetaAgent** (`alphabeta`): A negamax formulation of Minimax with alpha-beta pruning and move ordering, dramatically increasing search efficiency.
3. **MCTSAgent** (`mcts`): Pure Monte Carlo Tree Search. Uses random or Dijkstra rollouts to evaluate states. Balances exploration/exploitation via UCT.

## 4. Reinforcement Learning (RL) Pipeline

The RL system uses **Maskable Proximal Policy Optimization (MaskablePPO)** from the `sb3-contrib` library. It operates in the `solvers/rl/` package.

### State Representation
The agent receives a 3D tensor of shape `(9, 9, 6)`:
1. Player position (one-hot)
2. Opponent position (one-hot)
3. Horizontal walls (binary mask)
4. Vertical walls (binary mask)
5. Player distance heatmap (shortest path lengths)
6. Opponent distance heatmap

*The environment is symmetric; the agent always perceives itself as "Player 1" moving North.*

### Action Space & Masking
The action space is a discrete vector of length **140**:
- `0-3`: Step (N, E, S, W)
- `4-7`: Jump (over opponent)
- `8-11`: Diagonal Slide
- `12-75`: Horizontal walls
- `76-139`: Vertical walls

The core environment provides an **Action Mask** ensuring the neural network only updates probabilities for legal actions, completely bypassing the "invalid action penalty" problem.

### Neural Architectures
Two custom feature extractors are defined in `solvers/rl/feature_extractor.py`:
- `QuoridorCNN`: A standard 3-layer convolutional network mapping the 6 channels to a 256-D feature vector.
- `QuoridorResidualCNN`: A deeper architecture using residual blocks for advanced spatial reasoning.

### Training & Curriculum
The training script (`solvers/rl/train.py`) is driven by the `TrainConfig` dataclass. It supports automated **Curriculum Learning** (`solvers/rl/curriculum.py`):

```yaml
curriculum:
  - opponent: random
    timesteps: 100_000
    win_rate_threshold: 0.85
  - opponent: dijkstra
    timesteps: 300_000
    win_rate_threshold: 0.70
  - opponent: strategic
    timesteps: 500_000
    win_rate_threshold: 0.60
  - opponent: selfplay
    timesteps: 1_000_000
    win_rate_threshold: null
```
The curriculum automatically evaluates the agent against the current opponent and promotes it to the next stage when the `win_rate_threshold` is exceeded.
