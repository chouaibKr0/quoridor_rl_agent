# Quoridor RL Agent – V2 Implementation Plan

> **How to use this document.**  
> This plan is intentionally structured into **independent, reviewable stages**. Each stage produces a self-contained, testable result. The expectation is that an AI (Gemini) executes one stage at a time, the human reviews the output, optionally adjusts direction, and only then proceeds to the next stage. No stage assumes the next one will be executed immediately or unchanged.
>
> Stages are ordered by dependency: each stage can be started once the stages it depends on are ✅ complete. Stages within the same layer have no interdependency and could be done in any order or in parallel.

---

## Stage Map Overview

```
Stage 1 – Core Refactor (Foundation)
    ↓
Stage 2 – Solver Infrastructure (Plug-in system + heuristic solvers)
    ↓
Stage 3A – Search Solvers          Stage 3B – RL Solver Refactor
    ↓                                   ↓
Stage 4 – Experiment Runner (depends on 2, 3A, 3B)
    ↓
Stage 5 – Analysis Module (depends on 4)
    ↓
Stage 6 – UI / Replay + Final Integration
```

> **Human checkpoint**: After each stage, run the smoke tests listed at the bottom of that stage section before proceeding.

---

## Stage 1 — Core Refactor

**Goal**: Restructure V1's game engine into a clean `core/` package that all future modules depend on. Everything else in V1 continues to work identically after this stage.

**Depends on**: Nothing — this is the foundation.

**Motivation**: V1's game logic (`quoridor_game.py`) mixes `QuoridorGame`, `QuoridorStateBuilder`, and perspective-flip utilities in one file. V1's environment (`quoridor_env.py`) imports from that directly. This stage creates a stable, properly namespaced `core/` package before any new code is written.

### Files to Create

| File | Source | Action |
|---|---|---|
| `core/__init__.py` | — | New, empty (marks package) |
| `core/game.py` | `quoridor_game.py` | Move `QuoridorGame` + flip utilities |
| `core/state.py` | `quoridor_game.py` | Extract `QuoridorStateBuilder` only |
| `core/env.py` | `quoridor_env.py` | Move `QuoridorEnv` + `SelfPlayEnv` |
| `core/numba_utils.py` | `numba_utils.py` | Move as-is |

### Files to Modify

| File | Change |
|---|---|
| `core/game.py` | Update import of `QuoridorStateBuilder` → from `core.state`; remove deprecated `_has_path` stub |
| `core/env.py` | Update imports to use `core.game` |
| `main.py` | Update all imports to `core.*` |
| `train.py` | Update all imports to `core.*` |
| `evaluate.py` | Update all imports to `core.*` |
| `ai_agent.py` | Update imports to `core.*` (keep file itself unchanged for now; moved in Stage 2) |
| `feature_extractor.py` | Keep in root for now (moved in Stage 3B) |

### Specific Implementation Details

**`core/state.py`**: Extract `QuoridorStateBuilder` from `quoridor_game.py` verbatim. Add a module docstring. No logic changes.

**`core/game.py`**: Contains `QuoridorGame`, `flip_observation`, `flip_action`, `flip_mask`. Remove the `_has_path` method body entirely (was already `pass` + `DEPRECATED` comment). Add a module docstring.

**`core/env.py`**: Contains `QuoridorEnv` and `SelfPlayEnv`. Add a new optional constructor argument to `QuoridorEnv`:
```python
heatmaps_enabled: bool = True   # RQ1 ablation flag — added now, used in Stage 5
```
When `heatmaps_enabled=False`, Channels 4 and 5 of the observation tensor are zeroed out before being returned. This flag is wired in now so Stage 3B and Stage 5 can use it without touching `core/` again.

**Seed enforcement**: In `QuoridorGame.reset()`, accept an optional `seed: int = None` parameter and propagate it to numpy RNGs. Document that passing `seed` guarantees reproducibility.

### Files to Keep (Untouched)

- `quoridor_game.py` — kept as a **thin shim** that re-exports everything from `core.game` and `core.state` for backward compatibility with V1 trained model loading. Do not delete.
- `quoridor_env.py` — same: thin shim re-exporting from `core.env`.
- `ui.py` — untouched until Stage 6.

### ✅ Smoke Tests

```bash
# 1. Existing entry points still work
python main.py            # UI launches without import errors

# 2. Training still works
python train.py --opponent random --timesteps 500 --no-save

# 3. Evaluation still works  
python evaluate.py --model models/<any>.zip --opponent random --episodes 5

# 4. V1 shims work
python -c "from quoridor_game import QuoridorGame; g = QuoridorGame(); print('shim OK')"
python -c "from quoridor_env import QuoridorEnv; print('env shim OK')"

# 5. New core imports work
python -c "from core.game import QuoridorGame; from core.state import QuoridorStateBuilder; print('core OK')"
python -c "from core.env import QuoridorEnv, SelfPlayEnv; print('env OK')"
```

> **Human review point**: Verify all smoke tests pass. Check that `core/` file contents look clean and complete before proceeding. No new functionality yet — this is purely structural.

---

## Stage 2 — Solver Infrastructure & Heuristic Solvers

**Goal**: Establish the solver plugin system and migrate all existing heuristic agents into it. By the end of this stage, any code that needs an agent can call `get_solver("dijkstra", config)` and get a working agent.

**Depends on**: Stage 1 ✅

### Files to Create

| File | Content |
|---|---|
| `solvers/__init__.py` | Empty package marker |
| `solvers/base.py` | `BaseAgent` ABC (moved from `ai_agent.py`) |
| `solvers/registry.py` | Solver factory + registration system |
| `solvers/heuristic/__init__.py` | Empty |
| `solvers/heuristic/random_agent.py` | `RandomAgent` (moved from `ai_agent.py`) |
| `solvers/heuristic/dijkstra_agent.py` | `DijkstraAgent` (moved from `ai_agent.py`) |
| `solvers/heuristic/strategic_agent.py` | `StrategicAgent` (moved from `ai_agent.py`) |

### `solvers/base.py`

Move `BaseAgent` ABC exactly as in V1. Add one new abstract method for introspection:

```python
def get_config(self) -> dict:
    """Return solver hyperparameters as a flat dict (for logging)."""
    return {}
```

All existing agents implement this by returning their relevant constructor args (e.g., `{"wall_prob": self.wall_prob}`).

### `solvers/registry.py`

Central factory. Supports:
- **Registration**: `register_solver(name, cls)` — maps a string key to an agent class.
- **Instantiation**: `get_solver(name, config: dict) -> BaseAgent` — creates and returns an agent from a config dict.
- All heuristic solvers are auto-registered at import time via a module-level call.

```python
# Example usage
from solvers.registry import get_solver
agent = get_solver("strategic", {"wall_prob": 0.7, "seed": 42})
```

Config keys are solver-specific. Each agent class defines a `CONFIG_SCHEMA` class attribute listing accepted keys and defaults, used for validation and documentation.

### Migrating Existing Agents

Each agent is moved to its own file with these additions:
- Update imports to `from core.game import ...`, `from core.state import ...`
- Add `get_config()` implementation
- Add `CONFIG_SCHEMA` class attribute
- Add module-level `register_solver(...)` call

**Do not change any agent logic** — this is a pure migration.

### `ai_agent.py` Shim

After migration, replace `ai_agent.py` content with a shim:
```python
# Backward compatibility shim — V1 imports still work
from solvers.base import BaseAgent
from solvers.heuristic.random_agent import RandomAgent
from solvers.heuristic.dijkstra_agent import DijkstraAgent
from solvers.heuristic.strategic_agent import StrategicAgent
from solvers.registry import get_solver as get_agent
# MinimaxAgent shim added in Stage 3A
# RLAgent shim added in Stage 3B
```

### Modify `train.py` and `evaluate.py`

Update the `get_agent` import to come from `solvers.registry`. All behavior remains identical.

### ✅ Smoke Tests

```bash
# 1. Registry works
python -c "
from solvers.registry import get_solver
a = get_solver('random', {'seed': 0})
b = get_solver('dijkstra', {'seed': 0})
c = get_solver('strategic', {'wall_prob': 0.5, 'seed': 0})
print('All heuristic solvers OK:', a, b, c)
"

# 2. Old import path still works (shim)
python -c "from ai_agent import RandomAgent, DijkstraAgent, StrategicAgent; print('shim OK')"

# 3. Training with each heuristic opponent still works
python train.py --opponent random --timesteps 500 --no-save
python train.py --opponent dijkstra --timesteps 500 --no-save
python train.py --opponent strategic --timesteps 500 --no-save
```

> **Human review point**: Inspect `solvers/registry.py` and confirm the registry pattern is clean. Check `CONFIG_SCHEMA` on each agent is complete. This is the extensibility foundation — get it right before adding new solvers.

---

## Stage 3A — Search Solvers

**Goal**: Implement the three new tree-search solvers (improved Minimax, Alpha-Beta, pure MCTS), each with instrumentation hooks for RQ2 measurement.

**Depends on**: Stage 2 ✅  
**Independent of**: Stage 3B (can be done before, after, or in parallel)

### Files to Create

| File | Solver | Status |
|---|---|---|
| `solvers/search/__init__.py` | — | New |
| `solvers/search/minimax_agent.py` | Full Minimax with walls | Rewrite of V1 partial |
| `solvers/search/alphabeta_agent.py` | Alpha-Beta Negamax | New |
| `solvers/search/mcts_agent.py` | Pure MCTS | New |

### `solvers/search/minimax_agent.py` — Full Minimax

V1's `MinimaxAgent` only searched pawn moves. V2 extends it to include wall placements, with depth-limited search and a heuristic evaluation function.

**Key design decisions**:
- Evaluation function: `score = my_path_length - opponent_path_length` (same as V1, but now applied to wall-search nodes too).
- Wall action branching: To prevent search explosion, limit to the **top-K wall candidates** at each node. K is configurable (default: 10). Candidate selection: walls that increase opponent's path length by at least 1.
- `CONFIG_SCHEMA`: `{"depth": int, "wall_candidates_k": int, "seed": int}`
- Instrumentation: track `nodes_expanded` counter, reset each `select_action` call.

### `solvers/search/alphabeta_agent.py` — Alpha-Beta Negamax

Negamax formulation with alpha-beta pruning, move ordering (best pawn moves first, then wall candidates sorted by Δ*L* descending).

- Shares the same evaluation function as Minimax.
- `CONFIG_SCHEMA`: `{"depth": int, "wall_candidates_k": int, "seed": int}`
- Instrumentation: `nodes_expanded`, `pruned_nodes` counters.

### `solvers/search/mcts_agent.py` — Pure MCTS

Rollout-based MCTS (no learned policy). Standard UCT selection.

**Structure**:
- `MCTSNode`: stores state snapshot, visit count, total value, children.
- Rollout policy: random legal moves (can be upgraded to Dijkstra rollouts via config).
- Terminal detection: win/loss or max rollout depth.
- `CONFIG_SCHEMA`: `{"simulations": int, "rollout_policy": str, "c_puct": float, "seed": int}`
- Instrumentation: `nodes_expanded`, `simulations_run` counters per move.

> **Note on state snapshots**: MCTS requires cloning game state. `QuoridorGame` must support shallow copying. Add a `clone()` method to `core/game.py` in this stage: returns a deep copy of internal state without re-running Numba JIT.

### Instrumentation Interface

All search solvers expose:
```python
@property
def last_move_stats(self) -> dict:
    """Stats from the most recent select_action() call."""
    # e.g. {"nodes_expanded": 342, "pruned_nodes": 89, "elapsed_ms": 47.2}
```

This is used by the experiment runner (Stage 4) to log per-move compute stats for RQ2.

### Register in `ai_agent.py` Shim

Add to `ai_agent.py`:
```python
from solvers.search.minimax_agent import MinimaxAgent
```

### ✅ Smoke Tests

```bash
# 1. Each search solver instantiates and selects a legal action
python -c "
from core.game import QuoridorGame
from core.state import QuoridorStateBuilder
from solvers.search.minimax_agent import MinimaxAgent
from solvers.search.alphabeta_agent import AlphaBetaAgent
from solvers.search.mcts_agent import MCTSAgent

g = QuoridorGame()
obs = g._get_observation()
mask = g.get_legal_moves()

for AgentClass, cfg in [(MinimaxAgent, {'depth':2}), (AlphaBetaAgent, {'depth':2}), (MCTSAgent, {'simulations':20})]:
    a = AgentClass(**cfg)
    action = a.select_action(obs, mask)
    assert 0 <= action < 140, f'Invalid action from {AgentClass.__name__}'
    print(f'{AgentClass.__name__}: action={action}, stats={a.last_move_stats}')
print('All search solvers OK')
"

# 2. Registry entries work
python -c "
from solvers.registry import get_solver
get_solver('minimax', {'depth': 2})
get_solver('alphabeta', {'depth': 2})
get_solver('mcts', {'simulations': 50})
print('Registry OK')
"

# 3. Play 5-game match between Dijkstra and AlphaBeta (sanity check)
python evaluate.py --model models/<any>.zip --opponent random --episodes 5
```

> **Human review point**: Inspect minimax/MCTS implementations for correctness. Play a few AI vs AI games in the UI using the new solvers. Verify `last_move_stats` is populated. The quality of these solvers directly affects the validity of RQ2 — worth spending review time here.

---

## Stage 3B — RL Solver Refactor

**Goal**: Move the RL solver into the `solvers/rl/` package, make the training script config-driven, and add the automated curriculum pipeline.

**Depends on**: Stage 2 ✅  
**Independent of**: Stage 3A

### Files to Create / Move

| File | Source | Action |
|---|---|---|
| `solvers/rl/__init__.py` | — | New |
| `solvers/rl/ppo_agent.py` | `ai_agent.py` `RLAgent` | Move + extend |
| `solvers/rl/feature_extractor.py` | `feature_extractor.py` | Move as-is |
| `solvers/rl/train.py` | `train.py` | Refactor |
| `solvers/rl/curriculum.py` | — | New |

Keep `feature_extractor.py` and shim in `train.py` at root for backward compatibility.

### `solvers/rl/ppo_agent.py`

- Moves `RLAgent` from `ai_agent.py`.
- Adds `get_config()` returning `{"model_path": ..., "deterministic": ...}`.
- Adds `CONFIG_SCHEMA`.
- Registered in registry as `"ppo"`.

### `solvers/rl/train.py` — Config-Driven Refactor

The core change: `train()` accepts a **config dataclass** (or dict) instead of individual kwargs. This allows the experiment runner to call it programmatically.

```python
@dataclass
class TrainConfig:
    opponent: str = "random"
    total_timesteps: int = 1_000_000
    feature_extractor: str = "cnn"       # "cnn" | "residual"
    heatmaps_enabled: bool = True         # RQ1 ablation flag
    n_envs: int = 4
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    save_freq: int = 50_000
    log_dir: str = "./logs"
    model_dir: str = "./models"
    seed: int = 42
    load_model: str = None
    run_name: str = None   # auto-generated if None
```

The existing CLI (`python train.py --opponent ...`) still works — it parses args and builds a `TrainConfig`, then calls `train(config)`.

**New structured log**: At the end of training, save a `run_summary.json` alongside the model:
```json
{
  "run_name": "...",
  "config": { ... },
  "final_model_path": "...",
  "training_duration_s": 1234
}
```

### `solvers/rl/curriculum.py` — Automated Curriculum

Defines a curriculum as a sequence of `CurriculumStage` objects:

```python
@dataclass
class CurriculumStage:
    opponent: str
    timesteps: int
    win_rate_threshold: float   # advance to next stage when this is achieved
    eval_episodes: int = 50     # episodes used to measure win rate
    load_previous: bool = True  # load model from previous stage
```

A `CurriculumRunner` executes stages in sequence: after each stage's `timesteps`, evaluate win rate; if above threshold, proceed to next stage; otherwise, extend training by a configurable factor (default ×0.5 additional timesteps) up to a max of 2× original before advancing regardless.

Example curriculum config (YAML):
```yaml
curriculum:
  - opponent: random
    timesteps: 100000
    win_rate_threshold: 0.85
  - opponent: dijkstra
    timesteps: 300000
    win_rate_threshold: 0.70
  - opponent: strategic
    timesteps: 500000
    win_rate_threshold: 0.60
  - opponent: selfplay
    timesteps: 1000000
    win_rate_threshold: null   # final stage, no threshold
```

### ✅ Smoke Tests

```bash
# 1. Config-driven training works
python -c "
from solvers.rl.train import train, TrainConfig
cfg = TrainConfig(opponent='random', total_timesteps=300, n_envs=1, save=False)
train(cfg)
print('Config-driven training OK')
"

# 2. CLI still works
python train.py --opponent random --timesteps 300 --no-save

# 3. Heatmap ablation flag works
python -c "
from core.env import QuoridorEnv
from solvers.heuristic.random_agent import RandomAgent
env = QuoridorEnv(opponent=RandomAgent(), heatmaps_enabled=False)
obs, _ = env.reset()
assert obs[:,:,4].sum() == 0 and obs[:,:,5].sum() == 0, 'Heatmaps should be zeroed'
print('Heatmap ablation OK')
"

# 4. PPO agent loads and selects action
python -c "
from solvers.rl.ppo_agent import PPOAgent
a = PPOAgent('models/<any>.zip')
print('PPO agent OK, config:', a.get_config())
"

# 5. Root shim still works
python -c "from feature_extractor import QuoridorCNN; print('feature_extractor shim OK')"
```

> **Human review point**: Read `TrainConfig` and confirm all fields make sense. Inspect the `curriculum.py` design — the win rate threshold logic is the most likely area needing adjustment based on actual training behavior. The `heatmaps_enabled` flag is critical for RQ1 so confirm it zeros the right channels.

---

## Stage 4 — Experiment Runner

**Goal**: Build the config-driven experiment runner that orchestrates head-to-head matchups between any two registered solvers and saves structured results.

**Depends on**: Stage 2 ✅, Stage 3A ✅, Stage 3B ✅

### Files to Create

| File | Role |
|---|---|
| `experiments/__init__.py` | Package marker |
| `experiments/runner.py` | Core matchup executor |
| `experiments/config/` | Directory for YAML experiment configs |
| `experiments/config/example_heuristic_tournament.yaml` | Example |
| `experiments/config/example_rq1_ablation.yaml` | RQ1 ablation config |
| `experiments/config/example_rq2_search_budget.yaml` | RQ2 compute budget config |
| `experiments/results/` | Auto-created at runtime |

### `experiments/runner.py` — Experiment Config Schema

```python
@dataclass
class SolverSpec:
    type: str              # registry key, e.g. "mcts", "ppo", "dijkstra"
    config: dict = field(default_factory=dict)  # solver-specific kwargs

@dataclass  
class ExperimentConfig:
    experiment_name: str
    solver_a: SolverSpec
    solver_b: SolverSpec
    n_games: int = 100
    seed: int = 0
    output_dir: str = "experiments/results/"
    record_games: bool = True       # save action sequences as JSON
    record_telemetry: bool = True   # per-turn wall/path stats for RQ3
    latency_budget_ms: float = None # None = unlimited; set for RQ2
    swap_sides: bool = True         # play half games as each side
```

### Per-Game Record Format

Each game produces a record saved to `results/<experiment_name>/<game_id>.json`:

```json
{
  "game_id": 0,
  "solver_a": "mcts",
  "solver_b": "dijkstra",
  "winner": "a",
  "n_moves": 47,
  "moves": [12, 0, 34, ...],
  "telemetry": {
    "turn": [0, 1, 2, ...],
    "walls_a_cumulative": [0, 0, 0, 1, ...],
    "walls_b_cumulative": [0, 0, 1, 1, ...],
    "path_len_a": [8, 7, 7, ...],
    "path_len_b": [8, 8, 6, ...],
    "delta_L_a": [null, null, null, 3, ...],
    "delta_L_b": [null, null, 2, null, ...]
  },
  "per_move_stats": {
    "solver_a": [{"nodes_expanded": 342, "elapsed_ms": 47.2}, ...],
    "solver_b": [{"elapsed_ms": 0.1}, ...]
  }
}
```

`telemetry` is only recorded when `record_telemetry=True`. `per_move_stats` uses `last_move_stats` from search solvers (Stage 3A).

### Tournament Mode

When `experiment_name` contains multiple solver specs (list), the runner executes a round-robin bracket and produces an aggregate results file alongside individual game records.

### CLI Interface

```bash
# Run a single experiment from config file
python -m experiments.runner --config experiments/config/my_experiment.yaml

# Quick inline experiment (no config file)
python -m experiments.runner \
  --solver-a mcts simulations=200 \
  --solver-b dijkstra \
  --n-games 50 \
  --name "quick_mcts_vs_dijkstra"
```

### ✅ Smoke Tests

```bash
# 1. Run 5-game experiment from config file
python -m experiments.runner \
  --solver-a random \
  --solver-b dijkstra \
  --n-games 5 \
  --name "smoke_test" \
  --no-telemetry

# Check output exists
ls experiments/results/smoke_test/

# 2. Check record format
python -c "
import json, glob
f = glob.glob('experiments/results/smoke_test/*.json')[0]
rec = json.load(open(f))
assert 'winner' in rec and 'moves' in rec
print('Record format OK:', list(rec.keys()))
"

# 3. MCTS with latency budget
python -m experiments.runner \
  --solver-a "mcts simulations=500" \
  --solver-b dijkstra \
  --n-games 3 \
  --latency-budget-ms 200 \
  --name "smoke_latency"
```

> **Human review point**: Inspect a few game JSON records manually — confirm the telemetry data (path lengths, wall counts) looks plausible. This data feeds directly into RQ3 analysis. Also confirm latency budget enforcement is working as expected for RQ2.

---

## Stage 5 — Analysis Module

**Goal**: Build the analysis tooling that consumes experiment results and produces the metrics, comparisons, and visualizations needed to answer RQ1–RQ3.

**Depends on**: Stage 4 ✅ (needs experiment results format to be stable)

### Files to Create

| File | Role |
|---|---|
| `analysis/__init__.py` | Package marker |
| `analysis/metrics.py` | Per-game and aggregate metric computation |
| `analysis/rating.py` | TrueSkill / Elo rating system |
| `analysis/compare.py` | Multi-experiment comparison tables |
| `analysis/visualize.py` | Plot generation (matplotlib) |

### `analysis/metrics.py`

Loads game JSON records and computes:

| Metric | Used for |
|---|---|
| Win rate (per solver, with 95% CI via binomial) | All RQs |
| Mean / std game length | General |
| Wall expenditure curve (mean walls by turn) | RQ3 |
| Path-length delta per wall (Δ*L* distribution) | RQ3 |
| Entrapment turn (first turn Δ*L* > threshold) | RQ3 |
| Convergence curve (episode reward vs. timestep) | RQ1 |
| Nodes expanded per move (mean/p95) | RQ2 |
| Pruning ratio (for alpha-beta) | RQ2 |
| Latency budget utilization | RQ2 |

All metrics serializable to JSON for downstream use.

### `analysis/rating.py`

Implements TrueSkill-style Bayesian rating:
- Each solver has a `(mu, sigma)` rating.
- After each game result, update both solvers' ratings.
- Export final ratings table with confidence intervals: `mu ± 2*sigma`.
- Works on the full experiment results directory, or per-experiment.

Use the `trueskill` Python library if available; fall back to a simplified Elo implementation otherwise.

### `analysis/compare.py`

Reads multiple experiment result directories and produces:
- Markdown comparison table (solver vs. solver win rates + Elo).
- CSV export for further analysis.
- Summary of compute efficiency (nodes/s, latency utilization) for search solvers.

### `analysis/visualize.py`

Functions (each saves a `.png` to an output directory):

| Function | Plot | RQ |
|---|---|---|
| `plot_wall_spend_curves(results_dir)` | Mean cumulative walls by turn, per solver | RQ3 |
| `plot_delta_L_distribution(results_dir)` | Histogram of Δ*L* per wall placed | RQ3 |
| `plot_convergence(log_dir)` | Episode reward vs. timestep for RL runs | RQ1 |
| `plot_nodes_expanded(results_dir)` | Box plot of nodes expanded per move, per solver | RQ2 |
| `plot_elo_progression(results_dir)` | Elo ratings over tournament games | All |
| `plot_game_length_dist(results_dir)` | Histogram of game lengths per matchup | General |

### CLI Interface

```bash
# Compute metrics for an experiment
python -m analysis.metrics --results experiments/results/my_experiment/

# Generate all plots
python -m analysis.visualize --results experiments/results/my_experiment/ --out analysis/plots/

# Compare multiple experiments
python -m analysis.compare \
  experiments/results/exp_a/ \
  experiments/results/exp_b/ \
  --out analysis/reports/comparison.md
```

### ✅ Smoke Tests

```bash
# 1. Metrics load without errors on smoke test data
python -m analysis.metrics --results experiments/results/smoke_test/

# 2. Plots generate without errors
python -m analysis.visualize \
  --results experiments/results/smoke_test/ \
  --out /tmp/test_plots/
ls /tmp/test_plots/*.png

# 3. Compare two experiments
python -m analysis.compare \
  experiments/results/smoke_test/ \
  experiments/results/smoke_latency/ \
  --out /tmp/test_comparison.md
cat /tmp/test_comparison.md
```

> **Human review point**: Run a real small experiment (50–100 games) and inspect the generated plots. Confirm wall-spend curves and Δ*L* distributions look sensible. Adjust metric definitions if needed before running large experiments — changing metric definitions after the fact is expensive.

---

## Stage 6 — UI / Replay & Final Integration

**Goal**: Add game replay capability to the UI and ensure the full system is cleanly integrated end-to-end.

**Depends on**: All prior stages ✅

### Files to Create / Modify

| File | Action |
|---|---|
| `ui/__init__.py` | New package marker |
| `ui/ui.py` | Move from root `ui.py`, update imports |
| `ui/replay.py` | New: replay saved game JSON records |
| `main.py` | Update imports; add replay mode to menu |

### `ui/replay.py`

Loads a game JSON record (from Stage 4 experiment results) and:
- Re-renders the full game step-by-step in Pygame using the existing UI renderer.
- Supports pause, step forward, step backward, speed control.
- Displays per-move stats overlay (path lengths, walls remaining, solver move stats if available).

Interface:
```bash
# Replay a specific game
python main.py --replay experiments/results/my_experiment/game_0042.json
```

Or accessible from the existing main menu: "Replay Game → browse for file".

### Root `ui.py` Shim

Keep `ui.py` at root as shim re-exporting from `ui.ui` for backward compatibility.

### `main.py` Menu Updates

Add to the existing Pygame menu:
- **Replay Game** mode (opens file browser for `.json` game records).
- **AI vs AI** now shows a solver selection dropdown populated from the registry (all registered solver types).

### Final Integration Check

Run a full end-to-end pipeline pass:
1. Train a short PPO run → produces model.
2. Run experiment: PPO vs. MCTS (50 games) → produces results.
3. Run analysis → produces plots.
4. Replay one interesting game in the UI.

### ✅ Smoke Tests

```bash
# 1. Replay a game from experiment results
python main.py --replay experiments/results/smoke_test/game_0000.json

# 2. UI launches with updated menu
python main.py

# 3. Full pipeline end-to-end
python train.py --opponent random --timesteps 1000 --no-save
python -m experiments.runner \
  --solver-a random --solver-b dijkstra \
  --n-games 10 --name "e2e_test"
python -m analysis.metrics --results experiments/results/e2e_test/
python -m analysis.visualize --results experiments/results/e2e_test/ --out /tmp/e2e_plots/
echo "End-to-end pipeline OK"
```

> **Human review point**: Replay a real game and verify it looks correct. This is the final integration — confirm the full loop from training → experiment → analysis → replay is coherent and usable before declaring V2 complete.

---

## Dependency Summary

| Stage | Depends On | Can Be Skipped? | Blocking For |
|---|---|---|---|
| Stage 1 – Core Refactor | — | No — all other stages depend on it | Everything |
| Stage 2 – Solver Infrastructure | Stage 1 | No — registry is used everywhere | 3A, 3B, 4 |
| Stage 3A – Search Solvers | Stage 2 | No — needed for RQ2 | Stage 4 |
| Stage 3B – RL Refactor | Stage 2 | Partial — skip curriculum, keep basic PPO | Stage 4 |
| Stage 4 – Experiment Runner | 2, 3A, 3B | No — needed for any experiment | Stage 5 |
| Stage 5 – Analysis | Stage 4 | No — needed to answer RQs | Stage 6 |
| Stage 6 – UI Replay | All prior | Yes — replay is quality-of-life | Nothing |

---

## Open Decisions (Resolve Before Starting)

These are the open questions from the conceptual plan — answers are needed before Stage 1 begins.

1. **Package structure**: Recommend using `__init__.py` files (as designed in this plan) but **no `setup.py`** — keep it as a runnable project directory, not an installable package.
2. **Experiment config format**: Use **YAML** parsed into dataclasses. Best of both worlds: human-editable files, type-safe internal representation.
3. **AlphaZero stretch goal**: **Explicitly deferred to V3.** Stage 3A covers MCTS (pure) and Alpha-Beta only.
4. **Backward compatibility**: **Yes** — V1 root files become shims. V1 trained `.zip` models load identically via `solvers/rl/ppo_agent.py`.
5. **Result storage**: **Flat JSON files per game** (as designed in Stage 4). No database — simpler, inspectable, and sufficient for the experiment scale of V2.
