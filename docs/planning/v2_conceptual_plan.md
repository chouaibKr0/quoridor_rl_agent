# Quoridor RL Agent – V2 Conceptual Plan

> **Input to Implementation Planning.**  
> This document translates the V2 Manifesto into a concrete, structured conceptual plan grounded in a full reading of the V1 codebase. It defines *what* V2 is, *why* each part exists, and *how* the system should be organized — without prescribing specific implementation details yet.

---

## 1. What V1 Is (Baseline Understanding)

Before defining V2, it is important to understand what V1 achieved and where its boundaries are.

### V1 Architecture at a Glance

| Component | File | Role |
|---|---|---|
| Core Game Logic | `quoridor_game.py` | Board state, action decoding, reward, legal moves |
| RL Environment | `quoridor_env.py` | Gymnasium wrapper, action masking, self-play env |
| Feature Extraction | `feature_extractor.py` | CNN + ResidualCNN for 9×9×6 state |
| Agents / Solvers | `ai_agent.py` | Random, Dijkstra, Strategic, Minimax, RLAgent |
| Training | `train.py` | MaskablePPO, curriculum, vectorized envs |
| Evaluation | `evaluate.py` | Win rate stats against fixed opponents |
| UI | `ui.py` | Pygame interactive interface |
| Performance | `numba_utils.py` | Numba-JIT BFS, legal move computation |

### V1 Known Strengths
- Clean Gymnasium-compatible environment with correct action masking.
- Normalized perspective flip enabling proper self-play.
- Numba-accelerated BFS for real-time heatmaps and legal move computation.
- Modular agent ABC (`BaseAgent`) making it easy to add new solvers.
- Curriculum training pipeline: Random → Dijkstra → Strategic → Self-play.

### V1 Known Limitations (Motivating V2)
- **One algorithm**: Only MaskablePPO. No framework to plug in and compare multiple solvers.
- **No automated curriculum pipeline**: Training stages must be run manually in sequence.
- **No experiment tracking**: No structured way to record, compare, and analyze results across runs.
- **Evaluation is minimal**: Only win rate vs. a fixed opponent — no deeper analysis (game length, wall efficiency, opening behavior, etc.).
- **Agents are scattered**: Rule-based agents and RL agents live in the same file with no plugin structure.
- **MinimaxAgent is incomplete**: It only evaluates pawn moves and ignores wall placements in its search.
- **No reproducibility guarantees**: No standardized experiment configs, seeds not enforced consistently.
- **UI is decoupled from the research pipeline**: It can't be used to replay, visualize, or inspect experiments.

---

## 2. V2 Goals (From the Manifesto)

V2 has two intertwined purposes:

### 2.1 Learning by Practice
Implementing a diverse set of solvers to deeply understand each algorithmic family:
- **Reinforcement Learning**: PPO (V1), and new RL algorithms (DQN, AlphaZero-style MCTS+NN).
- **Classical Search**: Minimax (improve V1's partial implementation), MCTS (pure), Negamax, Iterative Deepening.
- **Heuristic / Greedy**: Dijkstra (V1), Strategic (V1), A*, mixed planners.

### 2.2 Research by Experimentation
Using Quoridor as a **"model organism"** — a minimal, fully discrete, two-player, zero-sum environment — to study principled decision-making under **adversarial topological disruption**: the opponent can dynamically reshape the graph of reachable states by placing walls, creating combinatorial action spaces and non-stationary shortest-path structures.

This framing shifts the research goal from "find the best solver" to **extracting generalizable methodological insights** about how different algorithm families (learned policy, tree search, heuristic planning) handle:
- Dynamic graph topology modified by the opponent in real time.
- Combinatorial resource management (finite wall budgets).
- The tension between short-term path optimization and long-term opponent containment.

---

## 3. V2 Architecture: High-Level Design

V2 reorganizes the project around three pillars: **Solvers**, **Experiments**, and **Analysis**.

```
quoridor_rl_agent/
│
├── core/                         # Stable, shared game engine (refactored V1)
│   ├── game.py                   # QuoridorGame (renamed from quoridor_game.py)
│   ├── env.py                    # Gymnasium environments (renamed from quoridor_env.py)
│   ├── state.py                  # QuoridorStateBuilder (extracted from game.py)
│   └── numba_utils.py            # Numba JIT utilities (unchanged or minor fixes)
│
├── solvers/                      # Plugin-style solver registry
│   ├── base.py                   # BaseAgent ABC (moved from ai_agent.py)
│   ├── registry.py               # Solver registry / factory
│   ├── heuristic/                # Rule-based agents
│   │   ├── random_agent.py
│   │   ├── dijkstra_agent.py
│   │   └── strategic_agent.py
│   ├── search/                   # Tree search agents
│   │   ├── minimax_agent.py      # Full minimax with wall moves (improved V1)
│   │   ├── mcts_agent.py         # Pure Monte Carlo Tree Search (new)
│   │   └── alphabeta_agent.py    # Alpha-beta pruning (new)
│   └── rl/                       # RL-based agents
│       ├── ppo_agent.py          # MaskablePPO wrapper (refactored V1 RLAgent)
│       ├── feature_extractor.py  # CNN / ResidualCNN (moved from root)
│       └── train.py              # RL training script (refactored V1 train.py)
│
├── experiments/                  # Experiment management
│   ├── config/                   # YAML/JSON experiment config files
│   │   └── example_experiment.yaml
│   ├── runner.py                 # Experiment runner (reads config, runs head-to-head)
│   └── results/                  # Auto-generated result files (JSON/CSV)
│
├── analysis/                     # Metrics, stats, and visualization
│   ├── metrics.py                # Extended metrics beyond win rate
│   ├── compare.py                # Multi-solver comparison reporting
│   └── visualize.py              # Plotting win rates, game length distributions, etc.
│
├── ui/                           # Interactive UI (refactored from ui.py)
│   ├── ui.py                     # Pygame rendering
│   └── replay.py                 # Replay saved games (new)
│
├── main.py                       # Entry point (unchanged behavior, updated imports)
└── evaluate.py                   # Standalone evaluation (kept, updated imports)
```

---

## 4. V2 Core Modules: What Changes and What Stays

### 4.1 Core Game Engine (`core/`) — Refactor Only
The game engine is solid and should not be redesigned. Changes are limited to:
- **Splitting** `quoridor_game.py`: Extract `QuoridorStateBuilder` into `core/state.py` for cleaner imports.
- **Renaming** files for clarity: `quoridor_game.py` → `core/game.py`, `quoridor_env.py` → `core/env.py`.
- **Fixing** the `DEPRECATED` `_has_path` method in `QuoridorGame`.
- **Ensuring full reproducibility**: enforce seed propagation across all game resets.

### 4.2 Solver Registry (`solvers/`) — Major Addition
This is the core new capability of V2.

- Each solver is a **self-contained module** implementing the `BaseAgent` interface.
- A central `registry.py` acts as a factory: `get_solver("mcts", config)` returns the appropriate agent.
- Solvers declare their own **config schema** (e.g., depth for minimax, simulations for MCTS, model path for PPO).
- New solvers to implement:
  - **Full Minimax with walls**: V1's minimax only evaluates pawn moves — extend to wall placement search.
  - **Pure MCTS**: Rollout-based Monte Carlo Tree Search without a learned policy.
  - **Alpha-Beta Negamax**: Proper negamax with alpha-beta pruning and move ordering.
  - **AlphaZero-style (stretch goal)**: MCTS guided by a learned value/policy network.

### 4.3 Experiment Runner (`experiments/`) — New Module
The key research infrastructure of V2.

- **Config-driven**: An experiment is fully defined by a YAML config specifying:
  - Solver A (type + hyperparameters)
  - Solver B (type + hyperparameters)
  - Number of games
  - Random seed
  - Output directory
- **Runner** executes head-to-head matchups, records outcomes per game.
- Results stored as structured JSON/CSV for downstream analysis.
- Supports **tournament mode**: run a round-robin bracket across N solvers.

Example experiment config:
```yaml
experiment_name: "ppo_vs_mcts_100games"
solver_a:
  type: ppo
  model_path: "models/quoridor_ppo_final_20260221.zip"
solver_b:
  type: mcts
  simulations: 500
  seed: 42
n_games: 100
seed: 0
output_dir: "experiments/results/"
```

### 4.4 Analysis Module (`analysis/`) — New Module
Extends V1's minimal evaluation to a full analysis suite. Metrics are designed to directly support RQ1–RQ3.

**Core metrics in `analysis/metrics.py`:**
- **Compute normalization**: All solvers are profiled under standardized latency budgets (e.g., 50 ms, 200 ms, 1 s per move) using node-expansion counters rather than raw game counts, ensuring fair comparison regardless of hardware speed.
- **Tournament Elo / TrueSkill**: A Bayesian rating system (TrueSkill or Glicko-2) with confidence intervals replaces raw pairwise win rates, enabling robust multi-solver rankings from relatively few games.
- **Topological telemetry** (for RQ3): Per-game, per-turn logging of:
  - Wall expenditure curve (walls placed by turn number).
  - Path-length displacement per wall placed (Δ*L* = opponent path length after wall − before wall).
  - Entrapment timing (turn at which opponent's path length exceeds a critical threshold).
- **Representation ablation support** (for RQ1): Environment flag to toggle BFS distance heatmaps (Channels 4–5) on/off, with convergence curve logging of episode reward and win rate per training step.
- **Search efficiency** (for RQ2): Node expansion counts and pruning ratios per move logged by `mcts_agent.py` and `minimax_agent.py` via instrumentation hooks.
- **Game length** distribution and **opening behavior** (first *N* moves) retained from original plan.

`compare.py` produces multi-solver comparison tables with Elo ratings and confidence intervals.  
`visualize.py` generates matplotlib/plotly plots: wall-spend curves, path-delta distributions, Elo progression over training.

### 4.5 RL Training Pipeline (`solvers/rl/train.py`) — Refactor
V1's `train.py` works but is monolithic. V2 refactors it to:
- Accept a config object (not just CLI args) so it can be called programmatically by the experiment runner.
- Support an **automated curriculum pipeline**: define stages in config, the trainer progresses automatically when win rate threshold is met.
- Log all hyperparameters and results to a structured file alongside TensorBoard.

### 4.6 UI and Replay (`ui/`) — Minor Addition
- Move existing `ui.py` into `ui/` package.
- Add `replay.py`: load a saved game (JSON of action sequences) and replay it visually. This enables inspection of interesting games found during experiments.

---

## 5. Research Questions V2 Should Be Able to Answer

Three cohesive, formally grounded research questions organized around the model organism paradigm:

| RQ | Formal Question | Core Focus | Required Pipeline Features |
|---|---|---|---|
| **RQ1** | *Topological Representation & Inductive Bias*: Can policy networks implicitly learn shortest-path topology directly from raw board layouts, or are hand-crafted BFS distance heatmaps essential for sample-efficient convergence? | Representation Learning | Toggleable BFS heatmaps (Channels 4–5) in `core/env.py`; convergence curve logging in `solvers/rl/train.py`; ablation configs in `experiments/config/`. |
| **RQ2** | *Combinatorial Action Pruning in Tree Search*: How effectively can a lightweight heuristic or learned prior filter non-viable wall actions to accelerate MCTS and Minimax under fixed latency budgets? | Search Optimization | Node expansion tracking and pruning hooks inside `solvers/search/mcts_agent.py` and `minimax_agent.py`; compute-normalized benchmarking in `analysis/metrics.py`. |
| **RQ3** | *Wall Economy & Resource Dynamics*: How do wall expenditure curves, path-extension deltas (Δ*L*), and entrapment timing evolve across self-play RL training versus heuristic opponents? | Game-Theoretic Behavior | Per-turn telemetry in `analysis/metrics.py`; wall-spend and Δ*L* logging in the experiment runner; visualization in `analysis/visualize.py`. |

---

## 6. What V2 Will NOT Do (Scope Boundaries)

- **No multi-player Quoridor** in V2 — classic 2-player only, as stated in the manifesto.
- **No online multiplayer / networking** — single-machine only.
- **No GUI redesign** — the Pygame UI is kept as-is with only replay additions.
- **No new game variants** (Ghost Quoridor, etc.) — deferred to V3+.
- **No external engine dependencies** — all solvers are implemented from scratch within this project; no external game engines, solver libraries, or cloud APIs.

> **Methodological note**: The goal of experiments is to extract *generalizable* design trade-offs — specifically the tension between deep policy intuition (learned representations) and deliberate lookahead search (tree search) in environments with dynamic network topology. All experimental execution remains strictly contained to the Quoridor environment, making results self-contained and reproducible without external dependencies.

---

## 7. Key Open Questions (For Discussion Before Implementation Planning)

1. **Package structure**: Should V2 use a proper Python package with `__init__.py` and `setup.py`, or keep the flat script structure of V1?
2. **Experiment config format**: YAML vs. Python dataclasses vs. JSON? YAML is human-readable; dataclasses give type safety.
3. **AlphaZero stretch goal**: Is this in scope for V2 or explicitly V3? It requires a significant self-play training loop separate from SB3.
4. **Backward compatibility**: Should V2 be able to load and run V1 trained models without changes?
5. **Result storage**: Flat CSV files vs. SQLite database for storing experiment results?
