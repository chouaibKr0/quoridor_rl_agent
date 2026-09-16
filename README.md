# Quoridor RL Agent (V2)

A comprehensive reinforcement learning and AI search framework that masters the board game [Quoridor](https://en.wikipedia.org/wiki/Quoridor). 

V2 introduces a modular architecture, a plug-and-play solver registry, automated curriculum learning, and a statistical experiment runner for evaluating AI agents.

## Overview

This project implements a complete AI pipeline for Quoridor:
- **Core Engine**: Fast, Numba-accelerated game rules and state management.
- **Gymnasium Environment**: Custom RL environment with intelligent action masking.
- **Multiple AI Solvers**: Heuristic, Tree-Search (Minimax, AlphaBeta, MCTS), and Neural Network (MaskablePPO) agents.
- **Experiment Framework**: Config-driven tournament system with rich telemetry (path lengths, nodes expanded).
- **Interactive UI**: Play against AI, watch AI vs AI, or replay saved game records step-by-step.

## Project Structure

The codebase is organized into modular packages:

```
quoridor_rl_agent/
├── core/                   # Game rules, state builder, and Gym environment
├── solvers/                # AI agents and the solver registry
│   ├── heuristic/          # Random, Dijkstra, Strategic
│   ├── search/             # Minimax, Alpha-Beta, MCTS
│   └── rl/                 # MaskablePPO, Curriculum, CNN extractors
├── experiments/            # Matchup runner and JSON telemetry recording
├── analysis/               # Metric computation, TrueSkill ratings, plotting
├── ui/                     # Pygame interface and game replay
├── docs/                   # Detailed documentation
└── models/                 # Saved RL neural network checkpoints
```

*For in-depth details on the architecture, solvers, and experiments, see the [docs/](docs/) directory.*

## Quick Start

### Play the Game
Launch the interactive Pygame UI:
```bash
python main.py
```
From the menu, you can select Human vs Human, Human vs AI, or AI vs AI. The solver dropdown automatically populates with all registered agents.

**Controls**:
- `M`: Move mode (click destination square)
- `W`: Wall mode (hover and click to place)
- `H`/`V`: Toggle wall orientation
- `R`: Reset game
- `ESC`: Return to menu

### Train the RL Agent
Train an agent against the `strategic` opponent for 500,000 timesteps:
```bash
python -m solvers.rl.train --opponent strategic --timesteps 500000
```
Or use the automated curriculum:
```bash
python -m solvers.rl.curriculum --config curriculum.yaml
```

### Run an AI Experiment
Run a head-to-head matchup between MCTS and Alpha-Beta:
```bash
python -m experiments.runner \
  --solver-a mcts simulations=500 \
  --solver-b alphabeta depth=2 \
  --n-games 50 \
  --name "mcts_vs_alphabeta"
```

### Analyze Results & Replay
Generate performance plots:
```bash
python -m analysis.visualize --results experiments/results/mcts_vs_alphabeta/ --out analysis/plots/
```
Replay a specific game in the UI:
```bash
python main.py --replay experiments/results/mcts_vs_alphabeta/game_0000.json
```

## Documentation

For deep-dives into specific systems, refer to the documentation in `docs/`:
- [System Architecture](docs/architecture.md)
- [Solvers & Training (RL, MCTS, Heuristics)](docs/solvers_and_training.md)
- [Experiments & Analysis Framework](docs/experiments_and_analysis.md)

## License

MIT
