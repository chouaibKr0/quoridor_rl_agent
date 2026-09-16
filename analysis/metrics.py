"""
Metrics Computation Module.
Loads game JSON records from experiments and computes statistical metrics,
95% confidence intervals, wall expenditure curves, delta_L distributions, and compute efficiency stats.
"""

import os
import json
import glob
import math
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np


def compute_binomial_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute Wilson score interval for binomial proportion (win rate).
    Returns (proportion, ci_lower, ci_upper).
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z = 1.95996  # 95% confidence z-score

    denominator = 1 + (z ** 2) / n
    centre_adjusted_probability = (p + (z ** 2) / (2 * n)) / denominator
    adjusted_standard_deviation = math.sqrt(
        (p * (1 - p) + (z ** 2) / (4 * n)) / n
    ) / denominator

    lower = max(0.0, centre_adjusted_probability - z * adjusted_standard_deviation)
    upper = min(1.0, centre_adjusted_probability + z * adjusted_standard_deviation)
    return (p, lower, upper)


def compute_experiment_metrics(results_dir: str) -> Dict[str, Any]:
    """Load all game_*.json files in results_dir and compute aggregate metrics."""
    game_files = sorted(glob.glob(os.path.join(results_dir, "game_*.json")))
    if not game_files:
        raise FileNotFoundError(f"No game_*.json files found in {results_dir}")

    records = []
    for fpath in game_files:
        with open(fpath, "r") as f:
            records.append(json.load(f))

    n_games = len(records)
    solver_a_name = records[0].get("solver_a", "solver_a")
    solver_b_name = records[0].get("solver_b", "solver_b")

    wins_a = sum(1 for r in records if r.get("winner") == "a")
    wins_b = sum(1 for r in records if r.get("winner") == "b")
    draws = sum(1 for r in records if r.get("winner") == "draw")

    p_a, ci_a_low, ci_a_high = compute_binomial_ci(wins_a, n_games)
    p_b, ci_b_low, ci_b_high = compute_binomial_ci(wins_b, n_games)

    game_lengths = [r.get("n_moves", 0) for r in records]

    # Telemetry aggregation
    walls_a_by_turn: Dict[int, List[int]] = {}
    walls_b_by_turn: Dict[int, List[int]] = {}
    delta_L_a_list: List[int] = []
    delta_L_b_list: List[int] = []

    nodes_a_list: List[int] = []
    nodes_b_list: List[int] = []
    pruned_b_list: List[int] = []

    for r in records:
        tel = r.get("telemetry", {})
        turns = tel.get("turn", [])
        w_a = tel.get("walls_a_cumulative", [])
        w_b = tel.get("walls_b_cumulative", [])
        dL_a = tel.get("delta_L_a", [])
        dL_b = tel.get("delta_L_b", [])

        for t, wa, wb in zip(turns, w_a, w_b):
            walls_a_by_turn.setdefault(t, []).append(wa)
            walls_b_by_turn.setdefault(t, []).append(wb)

        for d in dL_a:
            if d is not None:
                delta_L_a_list.append(d)
        for d in dL_b:
            if d is not None:
                delta_L_b_list.append(d)

        # Move stats
        stats = r.get("per_move_stats", {})
        for s in stats.get("solver_a", []):
            if "nodes_expanded" in s:
                nodes_a_list.append(s["nodes_expanded"])
        for s in stats.get("solver_b", []):
            if "nodes_expanded" in s:
                nodes_b_list.append(s["nodes_expanded"])
            if "pruned_nodes" in s:
                pruned_b_list.append(s["pruned_nodes"])

    # Compute wall curves (mean walls by turn)
    wall_curve_a = {t: float(np.mean(vals)) for t, vals in sorted(walls_a_by_turn.items())}
    wall_curve_b = {t: float(np.mean(vals)) for t, vals in sorted(walls_b_by_turn.items())}

    metrics = {
        "results_dir": results_dir,
        "n_games": n_games,
        "solver_a": solver_a_name,
        "solver_b": solver_b_name,
        "solver_a_metrics": {
            "wins": wins_a,
            "win_rate": p_a,
            "win_rate_ci_95": [ci_a_low, ci_a_high],
            "delta_L_distribution": {
                "mean": float(np.mean(delta_L_a_list)) if delta_L_a_list else 0.0,
                "max": int(np.max(delta_L_a_list)) if delta_L_a_list else 0,
                "counts": {str(k): int(v) for k, v in zip(*np.unique(delta_L_a_list, return_counts=True))} if delta_L_a_list else {},
            },
            "nodes_expanded": {
                "mean": float(np.mean(nodes_a_list)) if nodes_a_list else 0.0,
                "p95": float(np.percentile(nodes_a_list, 95)) if nodes_a_list else 0.0,
            },
        },
        "solver_b_metrics": {
            "wins": wins_b,
            "win_rate": p_b,
            "win_rate_ci_95": [ci_b_low, ci_b_high],
            "delta_L_distribution": {
                "mean": float(np.mean(delta_L_b_list)) if delta_L_b_list else 0.0,
                "max": int(np.max(delta_L_b_list)) if delta_L_b_list else 0,
                "counts": {str(k): int(v) for k, v in zip(*np.unique(delta_L_b_list, return_counts=True))} if delta_L_b_list else {},
            },
            "nodes_expanded": {
                "mean": float(np.mean(nodes_b_list)) if nodes_b_list else 0.0,
                "p95": float(np.percentile(nodes_b_list, 95)) if nodes_b_list else 0.0,
            },
            "pruned_nodes": {
                "mean": float(np.mean(pruned_b_list)) if pruned_b_list else 0.0,
            },
        },
        "draws": draws,
        "draw_rate": draws / n_games,
        "game_length": {
            "mean": float(np.mean(game_lengths)),
            "std": float(np.std(game_lengths)),
            "min": int(np.min(game_lengths)),
            "max": int(np.max(game_lengths)),
        },
        "wall_curves": {
            "solver_a": wall_curve_a,
            "solver_b": wall_curve_b,
        },
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Quoridor Experiment Metrics Computer")
    parser.add_argument("--results", type=str, required=True, help="Path to experiment results directory")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: metrics_summary.json in results dir)")

    args = parser.parse_args()

    m = compute_experiment_metrics(args.results)

    out_path = args.out or os.path.join(args.results, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Metrics Summary for: {m['results_dir']}")
    print(f"{'='*60}")
    print(f"Games: {m['n_games']}")
    print(f"Solver A ({m['solver_a']}): Win Rate = {m['solver_a_metrics']['win_rate']*100:.1f}% "
          f"[{m['solver_a_metrics']['win_rate_ci_95'][0]*100:.1f}%, {m['solver_a_metrics']['win_rate_ci_95'][1]*100:.1f}%]")
    print(f"Solver B ({m['solver_b']}): Win Rate = {m['solver_b_metrics']['win_rate']*100:.1f}% "
          f"[{m['solver_b_metrics']['win_rate_ci_95'][0]*100:.1f}%, {m['solver_b_metrics']['win_rate_ci_95'][1]*100:.1f}%]")
    print(f"Game Length: {m['game_length']['mean']:.1f} ± {m['game_length']['std']:.1f} moves")
    print(f"{'='*60}\n")
    print(f"Metrics exported to: {out_path}")


if __name__ == "__main__":
    main()
