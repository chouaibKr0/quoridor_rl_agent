# Experiments and Analysis

The Quoridor V2 framework includes an automated experiment runner and an analysis module designed to execute statistical matchups and visualize performance metrics.

## 1. Running Experiments

The `experiments/runner.py` module orchestrates automated matchups between any two solvers.

### Command Line Interface
You can run an inline experiment by providing the solver names directly:
```bash
python -m experiments.runner \
  --solver-a mcts simulations=200 \
  --solver-b alphabeta depth=2 \
  --n-games 100 \
  --name "mcts_vs_alphabeta"
```

### Config-Driven Tournaments
For complex setups, use YAML configuration files:
```yaml
# experiments/config/example_tournament.yaml
experiment_name: heuristic_tournament
n_games: 50
record_telemetry: true
solver_a:
  type: strategic
  config:
    wall_prob: 0.8
solver_b:
  type: dijkstra
```
Run with: `python -m experiments.runner --config experiments/config/example_tournament.yaml`

### JSON Records
Every game is saved as a JSON file in `experiments/results/<experiment_name>/`.
These records contain:
- Standard details: Winner, number of moves, and the action sequence.
- **Telemetry**: Cumulative walls spent, path length delta ($\Delta L$), and path lengths turn-by-turn.
- **Compute Stats**: Nodes expanded, pruned nodes, and elapsed time per move (for search solvers).

## 2. Analyzing Results

The `analysis/` package processes the JSON records.

### Metrics Computation
Generates win rates, average game lengths, nodes expanded, and resource utilization.
```bash
python -m analysis.metrics --results experiments/results/mcts_vs_alphabeta/
```

### TrueSkill / Elo Rating
Calculates ratings based on tournament matchups to statically rank the solvers.
```bash
python -m analysis.rating --results experiments/results/
```

### Multi-Experiment Comparison
Compare results across different experiment directories and generate markdown tables.
```bash
python -m analysis.compare experiments/results/exp1/ experiments/results/exp2/ --out reports/comparison.md
```

## 3. Visualization

The `analysis/visualize.py` module uses `matplotlib` to generate insight plots:

```bash
python -m analysis.visualize --results experiments/results/mcts_vs_alphabeta/ --out analysis/plots/
```

Generated plots include:
- **Wall Spend Curves**: Cumulative walls placed over the length of the game.
- **$\Delta L$ Distribution**: Histogram representing the effectiveness of wall placements.
- **Nodes Expanded**: Box plots showing the computational efficiency of the search solvers.
- **Convergence Curves**: Episode rewards over time (for RL training logs).
