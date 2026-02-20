# Quoridor RL Solver

A reinforcement learning agent that masters the board game [Quoridor](https://en.wikipedia.org/wiki/Quoridor) using Proximal Policy Optimization (PPO) with action masking.

## Overview

This project implements a complete RL training pipeline for Quoridor, featuring:
- **Custom Gymnasium environment** with intelligent action masking
- **CNN-based feature extraction** for spatial board understanding
- **Self-play training** with curriculum learning (3 opponent types)
- **Interactive UI** for human vs AI gameplay

## RL Approach

### Architecture
- **Algorithm**: MaskablePPO (PPO with invalid action masking)
- **Neural Network**: Custom CNN processing 9×9×6 state tensor
  - 6 channels: P1/P2 positions, H/V walls, distance heatmaps
- **Action Space**: 140 discrete actions (12 moves + 128 wall placements)

### Training Pipeline
```
Random Opponent → Dijkstra → Strategic → Self-Play
   (50k steps)     (200k)      (200k)     (500k+)
```

The agent learns progressively against increasingly sophisticated opponents, culminating in self-play for mastery.

The current implementation only support traing on only one agent at once (e,g,. trained on Dijkstra) thus no automated learnig pipline

### State Representation
The 9×9×6 observation tensor encodes:
1. Player 1 position (one-hot)
2. Player 2 position (one-hot)
3. Horizontal walls
4. Vertical walls
5. Player 1 distance heatmap (BFS-computed shortest paths)
6. Player 2 distance heatmap

This spatial representation enables the CNN to learn strategic wall placement and path planning.


## Usage

### Play Against AI
```bash
python main.py
```

The UI menu allows flexible matchmaking:
- **Human vs Human**: As an extra feature
- **Human vs AI**: Test your skills against an agents whther is it an RL agent or not
- **AI vs AI**: Watch different agents compete
- **RL Model Selection**: Choose from trained models or load custom checkpoints

**Controls**:
- `M`: Move mode (click destination square)
- `W`: Wall mode (hover and click to place)
- `H`/`V`: Toggle wall orientation
- `R`: Reset game
- `ESC`: Return to menu (during game)

### Train New Agent
```bash
# Train against Random opponent (baseline)
python train.py --opponent random --timesteps 50000

# Train against Dijkstra
python train.py --opponent dijkstra --timesteps 200000

# Train against Strategic
python train.py --opponent strategic --timesteps 500000

# Or Mixed traing (aginst both Dijkstra and Strategic)  with --opponent mixed

# Self-play training
python train.py --opponent selfplay --timesteps 500000


# Continue training from existing model (load model and resume training)
python train.py --load-model models/quoridor_ppo_final_20260207_133610.zip --opponent strategic --timesteps 100000
```

Models are saved to `models/` with naming convention: `models/quoridor_ppo_final_{date}_{time}.zip`

### Evaluate Agent
```bash
# Evaluate against specific opponent
python evaluate.py --model models/quoridor_ppo_final_{date}_{time}.zip --opponent dijkstra --episodes 100

# Evaluate against all opponents
python evaluate.py --model models/quoridor_ppo_final_{date}_{time}.zip --opponent all
```

## Project Structure

```
quoridor_rl_agent/
├── main.py                 # Interactive game UI with agent integration
├── quoridor_game.py        # Core game logic and state management
├── quoridor_env.py         # Gymnasium environment wrapper
├── feature_extractor.py    # Custom CNN architecture
├── ai_agent.py             # Baseline agents (Random, Dijkstra, Strategic...)
├── train.py                # Training script with curriculum learning
├── evaluate.py             # Agent evaluation utilities
├── ui.py                   # Pygame rendering and UI
└── models/                 # Trained model checkpoints
```

## Key Features

### Action Masking
The environment computes valid actions dynamically, masking illegal moves (occupied squares, blocked paths, invalid walls). This accelerates learning by preventing the agent from wasting exploration on invalid actions.

### Reward Shaping
```python
shaping = (0.1 * Δ_player_progress) + (0.05 * Δ_opponent_setback)
if player_wins:
  reword = 10.0 + shaping
elif player_losts:
  reword = -10.0 + shaping
else:
  reword = -0.01 + shaping
```

Encourages efficient play and offensive strategies.

### Opponent Agents
- **Random**: Uniform sampling over legal actions
- **Dijkstra**: Greedy shortest-path movement (no wall placement)
- **Strategic**: Dijkstra + opportunistic wall blocking

## License

MIT

