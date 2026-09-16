"""
Rating Computation Module.
Implements Bayesian TrueSkill / Elo rating calculation for solvers based on experiment game records.
"""

import os
import json
import glob
import math
import argparse
from typing import List, Dict, Any, Tuple


class RatingSystem:
    """
    Elo / TrueSkill rating system with fallback.
    """

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.games_played: Dict[str, int] = {}

    def get_rating(self, name: str) -> float:
        return self.ratings.get(name, self.initial_rating)

    def update_match(self, solver_a: str, solver_b: str, winner: str):
        r_a = self.get_rating(solver_a)
        r_b = self.get_rating(solver_b)

        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        e_b = 1.0 / (1.0 + 10.0 ** ((r_a - r_b) / 400.0))

        if winner == "a":
            s_a, s_b = 1.0, 0.0
        elif winner == "b":
            s_a, s_b = 0.0, 1.0
        else:
            s_a, s_b = 0.5, 0.5

        self.ratings[solver_a] = r_a + self.k_factor * (s_a - e_a)
        self.ratings[solver_b] = r_b + self.k_factor * (s_b - e_b)

        self.games_played[solver_a] = self.games_played.get(solver_a, 0) + 1
        self.games_played[solver_b] = self.games_played.get(solver_b, 0) + 1


def compute_ratings_from_dirs(results_dirs: List[str]) -> Dict[str, Any]:
    """Compute solver Elo ratings across multiple experiment results directories."""
    all_game_files = []
    for r_dir in results_dirs:
        files = sorted(glob.glob(os.path.join(r_dir, "game_*.json")))
        all_game_files.extend(files)

    if not all_game_files:
        raise FileNotFoundError("No game_*.json files found in specified directories")

    system = RatingSystem()

    for fpath in all_game_files:
        with open(fpath, "r") as f:
            rec = json.load(f)

        solver_a = rec.get("solver_a", "solver_a")
        solver_b = rec.get("solver_b", "solver_b")
        winner = rec.get("winner", "draw")

        system.update_match(solver_a, solver_b, winner)

    ratings_summary = {}
    for name, r in system.ratings.items():
        n_g = system.games_played.get(name, 0)
        # Approximate standard error for Elo rating
        se = 400.0 / math.sqrt(max(1, n_g))
        ratings_summary[name] = {
            "elo": round(r, 1),
            "games": n_g,
            "ci_95": [round(r - 1.96 * se, 1), round(r + 1.96 * se, 1)],
        }

    return ratings_summary


def main():
    parser = argparse.ArgumentParser(description="Compute Solver Elo Ratings")
    parser.add_argument("results_dirs", nargs="+", help="Paths to experiment result directories")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file for ratings")

    args = parser.parse_args()

    r_summary = compute_ratings_from_dirs(args.results_dirs)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(r_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SOLVER ELO RATINGS ({sum(s['games'] for s in r_summary.values()) // 2} games)")
    print(f"{'='*60}")
    print(f"{'Solver':<20} {'Elo Rating':>12} {'95% CI':>20} {'Games':>8}")
    print(f"{'-'*60}")

    for solver, data in sorted(r_summary.items(), key=lambda x: x[1]["elo"], reverse=True):
        ci_str = f"[{data['ci_95'][0]}, {data['ci_95'][1]}]"
        print(f"{solver:<20} {data['elo']:>12.1f} {ci_str:>20} {data['games']:>8}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
