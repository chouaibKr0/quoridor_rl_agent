# System Architecture

The Quoridor RL project is organized into modular packages to isolate the core game logic, AI agents, training pipelines, and experimental frameworks.

## High-Level Architecture

The system is split into the following major components:

1. **`core/`**: The immutable game engine and environment wrapper.
2. **`solvers/`**: The AI plugin system encompassing heuristics, search algorithms, and RL models.
3. **`experiments/`**: The orchestrator for head-to-head solver matches and data collection.
4. **`analysis/`**: Tools to parse experiment results, compute metrics, and generate visualizations.
5. **`ui/`**: Pygame-based UI for interactive play and game replay.

---

## 1. `core/` (Game Engine & State)

The `core` package is completely standalone. It knows nothing about AI or Pygame.

- `core.game.QuoridorGame`: The core rules engine. Manages board state, legal move generation (using Numba for speed), and win condition detection.
- `core.state.QuoridorStateBuilder`: Converts the internal game state into the `(9, 9, 6)` CNN observation tensor.
- `core.env.QuoridorEnv`: A Gymnasium (`gym.Env`) wrapper for RL training. It provides the standard `step()` and `reset()` API.
- `core.numba_utils.py`: Fast pathfinding and connectivity checks for the distance heatmaps.

---

## 2. `solvers/` (Agent Registry)

All AI agents are "solvers". A registry pattern makes them interchangeable in the UI and experiment runner.

- `solvers.registry`: The central factory (`register_solver`, `get_solver`).
- `solvers.base.BaseAgent`: The abstract base class all solvers must implement (`select_action`, `get_config`).

### Solver Subpackages
- **`solvers/heuristic/`**: Hardcoded logic agents (`RandomAgent`, `DijkstraAgent`, `StrategicAgent`).
- **`solvers/search/`**: Tree-search agents (`MinimaxAgent`, `AlphaBetaAgent`, `MCTSAgent`).
- **`solvers/rl/`**: Neural network models (`PPOAgent`, `feature_extractor.py`) and training orchestration (`train.py`, `curriculum.py`).

---

## 3. `experiments/` (Evaluation Framework)

The experiment runner automates thousands of games to compare solvers scientifically.

- **`experiments.runner`**: Executes configured matchups between any two registered solvers. Outputs detailed JSON records containing action sequences, solver telemetry (e.g. nodes expanded), and game telemetry (e.g. path lengths).
- Configured via YAML files in `experiments/config/`.

---

## 4. `analysis/` (Metrics & Visualization)

Consumes the JSON output from `experiments/runner`.

- `analysis.metrics`: Computes win rates, path-length deltas ($\Delta L$), and compute utilization (RQ1–RQ3).
- `analysis.rating`: Computes TrueSkill/Elo ratings across a tournament.
- `analysis.compare`: Generates markdown comparisons of solvers.
- `analysis.visualize`: Produces matplotlib charts (wall spend curves, $\Delta L$ distribution, nodes expanded box plots).

---

## 5. `ui/` (Presentation Layer)

- `ui.ui.QuoridorUI`: The main Pygame application for interactive play.
- `ui.replay.GameReplayer`: A specialized mode that loads an experiment JSON record and allows step-by-step playback of an AI game with rich telemetry overlays.

*Note: For backward compatibility with V1, shim modules exist in the project root (e.g., `ai_agent.py`, `quoridor_game.py`) to maintain compatibility with older trained checkpoints.*
