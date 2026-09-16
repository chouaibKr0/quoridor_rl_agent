# Quoridor RL Agent: AI Architecture and Pipeline Documentation

This document describes the underlying AI architecture, agent designs, and training pipeline of the Quoridor Reinforcement Learning (RL) project.

---

## 1. Overall AI Architecture

The project relies on Reinforcement Learning (RL) and specifically utilizes the **Maskable Proximal Policy Optimization (MaskablePPO)** algorithm from the `sb3-contrib` library. Quoridor is a complex game with a large and dynamic action space, meaning many actions are invalid at any given state. MaskablePPO allows the model to only select from valid (legal) moves using an action mask provided by the environment.

### 1.1 State Representation (Observation Space)
The game state is represented as a 3D tensor of shape `(9, 9, 6)`. The 6 channels encode essential information:
1. **Channel 0**: Player 1 position (one-hot matrix).
2. **Channel 1**: Player 2 position (one-hot matrix).
3. **Channel 2**: Horizontal walls.
4. **Channel 3**: Vertical walls.
5. **Channel 4**: Player 1 distance heatmap (shortest path to goal).
6. **Channel 5**: Player 2 distance heatmap (shortest path to goal).

*Note: The environment implements a fully normalized perspective, meaning the RL agent always perceives itself as "Player 1" moving towards the top of the board.*

### 1.2 Action Space
The action space is a discrete vector of length **140**, partitioned as follows:
- `0-3`: Step (North, East, South, West)
- `4-7`: Jump (North, East, South, West - over opponent)
- `8-11`: Slide (NE, NW, SE, SW - diagonal moves)
- `12-75`: Horizontal walls (64 possible positions)
- `76-139`: Vertical walls (64 possible positions)

An **Action Mask** is provided by the underlying game engine to restrict the model from predicting illegal actions (like moving out of bounds, placing a wall overlapping another, or completely blocking a player's path).

### 1.3 Feature Extraction (Neural Networks)
To process the spatial dependencies of the `9x9x6` input, the framework implements custom PyTorch Convolutional Neural Networks (CNNs).
1. **QuoridorCNN**: A standard 3-layer CNN mapping the channels to a 256-dimensional feature vector.
2. **QuoridorResidualCNN**: A deeper architecture employing residual blocks (ResNet style) to capture more intricate patterns of wall placement and path-finding.

---

## 2. Environments

The training framework uses OpenAI Gymnasium customized environments:
- **`QuoridorEnv`**: An environment where the RL agent plays against a specific, fixed heuristic or pre-trained opponent. 
- **`SelfPlayEnv`**: An advanced training environment where the RL agent plays against a copy of itself. The state gets flipped automatically so each player receives observations from a standardized perspective.

---

## 3. Agent Baselines and Opponents

To evaluate performance and bootstrap training, several rule-based heuristic agents are implemented:
1. **`RandomAgent`**: Selects a uniformly random action from the set of valid moves. Acts as a baseline.
2. **`DijkstraAgent`**: A greedy agent that uses the distance heatmaps to always take the pawn step that minimizes its path to the goal. It never places walls.
3. **`StrategicAgent`**: Evaluates available wall placements to maximize the opponent's path length. If no wall significantly restricts the opponent, it defaults to the Dijkstra strategy to progress towards its own goal.
4. **`MinimaxAgent`**: Evaluates moves using the Minimax algorithm with alpha-beta pruning. It uses the difference in path length between itself and the opponent as a scoring heuristic.
5. **`RLAgent`**: The primary neural network-based agent that loads a trained `MaskablePPO` policy.

---

## 4. Training Pipeline

The training pipeline is controlled by the `train.py` script. The pipeline involves:

1. **Environment Vectorization**: The training script wraps multiple `QuoridorEnv` (or `SelfPlayEnv`) instances inside a `SubprocVecEnv` (or `DummyVecEnv` for smaller setups) to collect experience in parallel, speeding up the data collection process for PPO.
2. **Model Definition**: A `MaskablePPO` agent is initialized equipped with a generic policy network and the user-specified CNN feature extractor.
3. **Checkpoints and Monitoring**: The training loop relies on stable-baseline3 callbacks to periodically save `.zip` checkpoints. It also integrates Tensorboard logging to track metrics like episodic reward, loss, and entropy.
4. **Curriculum / Self-Play**: Training can handle multiple curriculum styles:
   - *Fixed Opponent Training*: The model learns to consistently beat heuristic baselines (e.g., `dijkstra` or `strategic`).
   - *Self-Play*: The model continuously bootstraps its skill by playing against itself, ensuring it discovers diverse strategies without overfitting to a static heuristic.
5. **Resuming Training**: The pipeline seamlessly allows loading a previously saved `.zip` model and continuing training, automatically applying the specified learning rate schedule.

### 4.1 Example Training Execution
```bash
python train.py --opponent selfplay --extractor residual --n-envs 8 --timesteps 2000000 --save-freq 50000
```
This spawns 8 parallel environments utilizing the residual network feature extractor to train via self-play for 2 million steps.
