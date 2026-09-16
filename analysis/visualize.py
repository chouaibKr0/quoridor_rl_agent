"""
Visualization Module.
Generates publication-quality Matplotlib charts for RQ1 (convergence/ablation),
RQ2 (search node efficiency), and RQ3 (wall expenditure & delta_L distributions).
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import compute_experiment_metrics


def plot_wall_spend_curves(results_dir: str, out_dir: str):
    """Plot mean cumulative walls placed vs turn number (RQ3)."""
    metrics = compute_experiment_metrics(results_dir)
    curves = metrics["wall_curves"]

    turns_a = sorted([int(k) for k in curves["solver_a"].keys()])
    walls_a = [curves["solver_a"].get(t, curves["solver_a"].get(str(t), 0)) for t in turns_a]

    turns_b = sorted([int(k) for k in curves["solver_b"].keys()])
    walls_b = [curves["solver_b"].get(t, curves["solver_b"].get(str(t), 0)) for t in turns_b]

    plt.figure(figsize=(8, 5))
    plt.plot(turns_a, walls_a, label=f"{metrics['solver_a']}", color="#1f77b4", linewidth=2)
    plt.plot(turns_b, walls_b, label=f"{metrics['solver_b']}", color="#ff7f0e", linewidth=2, linestyle="--")

    plt.title(f"Wall Expenditure Curve — {metrics['solver_a']} vs {metrics['solver_b']}")
    plt.xlabel("Turn Number")
    plt.ylabel("Mean Cumulative Walls Placed")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wall_spend_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_delta_L_distribution(results_dir: str, out_dir: str):
    """Plot histogram of path length increase delta_L per wall placed (RQ3)."""
    metrics = compute_experiment_metrics(results_dir)
    dist_a = metrics["solver_a_metrics"]["delta_L_distribution"].get("counts", {})
    dist_b = metrics["solver_b_metrics"]["delta_L_distribution"].get("counts", {})

    all_keys = sorted(list(set([int(k) for k in dist_a.keys()] + [int(k) for k in dist_b.keys()])))
    if not all_keys:
        all_keys = [1, 2, 3]

    counts_a = [dist_a.get(str(k), 0) for k in all_keys]
    counts_b = [dist_b.get(str(k), 0) for k in all_keys]

    x = np.arange(len(all_keys))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, counts_a, width, label=metrics['solver_a'], color="#1f77b4")
    plt.bar(x + width / 2, counts_b, width, label=metrics['solver_b'], color="#ff7f0e")

    plt.title(f"Path Length Delta (ΔL) Distribution Per Wall")
    plt.xlabel("Opponent Path Length Increase (ΔL)")
    plt.ylabel("Occurrences")
    plt.xticks(x, [f"+{k}" for k in all_keys])
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "delta_L_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_nodes_expanded(results_dir: str, out_dir: str):
    """Plot nodes expanded per move (RQ2)."""
    metrics = compute_experiment_metrics(results_dir)
    m_a = metrics["solver_a_metrics"]["nodes_expanded"]
    m_b = metrics["solver_b_metrics"]["nodes_expanded"]

    solvers = [metrics["solver_a"], metrics["solver_b"]]
    means = [m_a["mean"], m_b["mean"]]
    p95s = [m_a["p95"], m_b["p95"]]

    x = np.arange(len(solvers))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, means, width, label="Mean Nodes", color="#2ca02c")
    plt.bar(x + width / 2, p95s, width, label="95th Percentile Nodes", color="#d62728")

    plt.title("Search Nodes Expanded Per Move (Compute Efficiency)")
    plt.ylabel("Nodes Expanded")
    plt.xticks(x, solvers)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nodes_expanded.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_game_length_dist(results_dir: str, out_dir: str):
    """Histogram of game lengths in moves."""
    game_files = sorted(glob.glob(os.path.join(results_dir, "game_*.json")))
    lengths = []
    for f in game_files:
        with open(f, "r") as fp:
            lengths.append(json.load(fp).get("n_moves", 0))

    plt.figure(figsize=(7, 5))
    plt.hist(lengths, bins=15, color="#9467bd", edgecolor="black", alpha=0.7)
    plt.title("Game Length Distribution (Total Moves per Game)")
    plt.xlabel("Moves")
    plt.ylabel("Game Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "game_length_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_all_plots(results_dir: str, out_dir: str):
    """Run all plot generators for an experiment."""
    plot_wall_spend_curves(results_dir, out_dir)
    plot_delta_L_distribution(results_dir, out_dir)
    plot_nodes_expanded(results_dir, out_dir)
    plot_game_length_dist(results_dir, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Analysis Plots")
    parser.add_argument("--results", type=str, required=True, help="Path to experiment results directory")
    parser.add_argument("--out", type=str, default="analysis/plots/", help="Output directory for plots")

    args = parser.parse_args()
    generate_all_plots(args.results, args.out)


if __name__ == "__main__":
    main()
