"""
Multi-Experiment Comparison Module.
Reads multiple experiment directories and generates Markdown comparison tables and CSV exports.
"""

import os
import json
import glob
import csv
import argparse
from typing import List, Dict, Any

from analysis.metrics import compute_experiment_metrics
from analysis.rating import compute_ratings_from_dirs


def generate_comparison_report(results_dirs: List[str], output_markdown_path: str, output_csv_path: str = None):
    """Generate Markdown and CSV comparative report across experiments."""
    all_metrics = []
    for r_dir in results_dirs:
        m = compute_experiment_metrics(r_dir)
        all_metrics.append(m)

    ratings = compute_ratings_from_dirs(results_dirs)

    # Markdown output construction
    md_lines = [
        "# Quoridor V2 — Experiment Comparison Report",
        "",
        "## Overall Solver Elo Ratings",
        "",
        "| Solver | Elo Rating | 95% Confidence Interval | Total Games Played |",
        "|---|---|---|---|",
    ]

    for solver, data in sorted(ratings.items(), key=lambda x: x[1]["elo"], reverse=True):
        ci_str = f"[{data['ci_95'][0]}, {data['ci_95'][1]}]"
        md_lines.append(f"| `{solver}` | **{data['elo']:.1f}** | {ci_str} | {data['games']} |")

    md_lines.extend([
        "",
        "## Head-to-Head Experiment Results",
        "",
        "| Experiment | Solver A | Solver B | Win Rate A | Win Rate B | Draw Rate | Mean Moves |",
        "|---|---|---|---|---|---|---|",
    ])

    csv_rows = [["experiment", "solver_a", "solver_b", "win_rate_a", "win_rate_b", "draw_rate", "mean_moves"]]

    for m in all_metrics:
        exp_name = os.path.basename(os.path.normpath(m["results_dir"]))
        sa = m["solver_a"]
        sb = m["solver_b"]
        wr_a = f"{m['solver_a_metrics']['win_rate']*100:.1f}%"
        wr_b = f"{m['solver_b_metrics']['win_rate']*100:.1f}%"
        dr = f"{m['draw_rate']*100:.1f}%"
        mean_m = f"{m['game_length']['mean']:.1f} ± {m['game_length']['std']:.1f}"

        md_lines.append(f"| `{exp_name}` | `{sa}` | `{sb}` | {wr_a} | {wr_b} | {dr} | {mean_m} |")
        csv_rows.append([exp_name, sa, sb, m['solver_a_metrics']['win_rate'], m['solver_b_metrics']['win_rate'], m['draw_rate'], m['game_length']['mean']])

    os.makedirs(os.path.dirname(os.path.abspath(output_markdown_path)), exist_ok=True)
    with open(output_markdown_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

    print(f"\nComparison report generated:")
    print(f"  Markdown: {output_markdown_path}")
    if output_csv_path:
        print(f"  CSV: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Experiment Comparison Report")
    parser.add_argument("results_dirs", nargs="+", help="Paths to experiment result directories")
    parser.add_argument("--out", type=str, required=True, help="Output Markdown report path")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV export path")

    args = parser.parse_args()
    generate_comparison_report(args.results_dirs, args.out, args.csv)


if __name__ == "__main__":
    main()
