"""
Experiment Runner Engine.
Orchestrates head-to-head matchups between Quoridor solvers with full telemetry
and latency budget enforcement for RQ1-RQ3 evaluation.
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np

from core.game import QuoridorGame, flip_observation, flip_action, flip_mask
from solvers.registry import get_solver, get_agent
from solvers.base import BaseAgent


@dataclass
class SolverSpec:
    """Specification of a solver and its configuration."""
    type: str                                    # Registry key (e.g. "mcts", "ppo", "dijkstra")
    config: Dict[str, Any] = field(default_factory=dict)  # Hyperparameters


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_name: str
    solver_a: SolverSpec
    solver_b: SolverSpec
    n_games: int = 100
    seed: int = 0
    output_dir: str = "experiments/results/"
    record_games: bool = True
    record_telemetry: bool = True
    latency_budget_ms: Optional[float] = None
    swap_sides: bool = True


def parse_solver_spec_string(spec_str: str) -> SolverSpec:
    """Parse solver spec from command line argument, e.g. 'mcts simulations=200 c_puct=1.4'."""
    tokens = spec_str.strip().split()
    if not tokens:
        raise ValueError("Empty solver spec string")
    solver_type = tokens[0]
    config = {}
    for token in tokens[1:]:
        if "=" in token:
            key, val = token.split("=", 1)
            # Try type coercion: int -> float -> bool -> str
            if val.lower() == "true":
                config[key] = True
            elif val.lower() == "false":
                config[key] = False
            else:
                try:
                    config[key] = int(val)
                except ValueError:
                    try:
                        config[key] = float(val)
                    except ValueError:
                        config[key] = val
    return SolverSpec(type=solver_type, config=config)


def run_single_game(
    game_id: int,
    config: ExperimentConfig,
    agent_a_spec: SolverSpec,
    agent_b_spec: SolverSpec,
    a_is_p1: bool,
) -> Dict[str, Any]:
    """Execute a single head-to-head match and return structured telemetry record."""
    game_seed = config.seed + game_id
    game = QuoridorGame()
    game.reset(seed=game_seed)

    # Instantiate agents
    p1_spec = agent_a_spec if a_is_p1 else agent_b_spec
    p2_spec = agent_b_spec if a_is_p1 else agent_a_spec

    p1_cfg = dict(p1_spec.config)
    p1_cfg.update({"player": 1, "seed": game_seed})
    p2_cfg = dict(p2_spec.config)
    p2_cfg.update({"player": 2, "seed": game_seed + 1000})

    agent_p1 = get_solver(p1_spec.type, config=p1_cfg)
    agent_p2 = get_solver(p2_spec.type, config=p2_cfg)

    moves_history: List[int] = []
    move_stats_history: Dict[str, List[Dict[str, Any]]] = {"solver_a": [], "solver_b": []}

    # Telemetry tracking
    telemetry: Dict[str, List[Any]] = {
        "turn": [],
        "walls_a_cumulative": [],
        "walls_b_cumulative": [],
        "path_len_a": [],
        "path_len_b": [],
        "delta_L_a": [],
        "delta_L_b": [],
    }

    turn = 0
    max_steps = 200

    while not game.done and turn < max_steps:
        current_p1 = (game.current_player == 1)
        current_agent_label = "solver_a" if (current_p1 == a_is_p1) else "solver_b"
        current_agent = agent_p1 if current_p1 else agent_p2

        obs = game._get_observation()
        mask = game.get_legal_moves()

        # Perspective flip if Player 2
        if not current_p1:
            obs_eval = flip_observation(obs)
            mask_eval = flip_mask(mask)
        else:
            obs_eval = obs
            mask_eval = mask

        start_t = time.perf_counter()
        raw_action = current_agent.select_action(obs_eval, mask_eval)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Enforce latency budget if active
        budget_exceeded = False
        if config.latency_budget_ms is not None and elapsed_ms > config.latency_budget_ms:
            budget_exceeded = True
            # Fallback to random pawn move if over budget
            valid_pawn = [a for a in np.where(mask_eval > 0)[0] if a < 12]
            if valid_pawn:
                raw_action = valid_pawn[0]

        action = flip_action(raw_action) if not current_p1 else raw_action

        # Collect move stats
        m_stats = {"elapsed_ms": round(elapsed_ms, 3), "budget_exceeded": budget_exceeded}
        if hasattr(current_agent, "last_move_stats") and isinstance(current_agent.last_move_stats, dict):
            m_stats.update(current_agent.last_move_stats)
        move_stats_history[current_agent_label].append(m_stats)

        # Pre-move path lengths for telemetry
        p1_path_prev = game._get_path_len(game.p1_pos, obs[:, :, 4])
        p2_path_prev = game._get_path_len(game.p2_pos, obs[:, :, 5])

        # Step game
        game.step(action)
        moves_history.append(int(action))

        # Telemetry recording
        if config.record_telemetry:
            next_obs = game._get_observation()
            p1_path_curr = game._get_path_len(game.p1_pos, next_obs[:, :, 4])
            p2_path_curr = game._get_path_len(game.p2_pos, next_obs[:, :, 5])

            walls_p1 = 10 - game.p1_walls_left
            walls_p2 = 10 - game.p2_walls_left

            walls_a = walls_p1 if a_is_p1 else walls_p2
            walls_b = walls_p2 if a_is_p1 else walls_p1
            path_a = p1_path_curr if a_is_p1 else p2_path_curr
            path_b = p2_path_curr if a_is_p1 else p1_path_curr

            delta_l_a = None
            delta_l_b = None

            if action >= 12:
                # Wall action placed this turn
                if current_p1:
                    delta_l = p2_path_curr - p2_path_prev
                    if a_is_p1:
                        delta_l_a = delta_l
                    else:
                        delta_l_b = delta_l
                else:
                    delta_l = p1_path_curr - p1_path_prev
                    if a_is_p1:
                        delta_l_b = delta_l
                    else:
                        delta_l_a = delta_l

            telemetry["turn"].append(turn)
            telemetry["walls_a_cumulative"].append(walls_a)
            telemetry["walls_b_cumulative"].append(walls_b)
            telemetry["path_len_a"].append(path_a)
            telemetry["path_len_b"].append(path_b)
            telemetry["delta_L_a"].append(delta_l_a)
            telemetry["delta_L_b"].append(delta_l_b)

        turn += 1

    # Winner determination
    if game.p1_pos[0] == 8:
        p1_won = True
        p2_won = False
    elif game.p2_pos[0] == 0:
        p1_won = False
        p2_won = True
    else:
        p1_won = False
        p2_won = False

    if p1_won:
        winner = "a" if a_is_p1 else "b"
    elif p2_won:
        winner = "b" if a_is_p1 else "a"
    else:
        winner = "draw"

    record = {
        "game_id": game_id,
        "solver_a": agent_a_spec.type,
        "solver_b": agent_b_spec.type,
        "solver_a_config": agent_a_spec.config,
        "solver_b_config": agent_b_spec.config,
        "a_is_player_1": a_is_p1,
        "winner": winner,
        "n_moves": len(moves_history),
        "moves": moves_history,
        "per_move_stats": move_stats_history,
    }

    if config.record_telemetry:
        record["telemetry"] = telemetry

    return record


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run full head-to-head experiment series and save output records."""
    exp_dir = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save experiment configuration metadata
    with open(os.path.join(exp_dir, "experiment_config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*70}")
    print(f"Running Experiment: {config.experiment_name}")
    print(f"Solver A: {config.solver_a.type} ({config.solver_a.config})")
    print(f"Solver B: {config.solver_b.type} ({config.solver_b.config})")
    print(f"Total Games: {config.n_games} | Swap Sides: {config.swap_sides}")
    print(f"{'='*70}\n")

    wins_a = 0
    wins_b = 0
    draws = 0

    start_time = time.time()

    for g_id in range(config.n_games):
        a_is_p1 = not (config.swap_sides and (g_id % 2 == 1))
        rec = run_single_game(g_id, config, config.solver_a, config.solver_b, a_is_p1)

        w = rec["winner"]
        if w == "a":
            wins_a += 1
        elif w == "b":
            wins_b += 1
        else:
            draws += 1

        if config.record_games:
            game_file = os.path.join(exp_dir, f"game_{g_id:04d}.json")
            with open(game_file, "w") as f:
                json.dump(rec, f, indent=2)

        if (g_id + 1) % max(1, config.n_games // 5) == 0 or (g_id + 1) == config.n_games:
            print(f"  Game {g_id + 1}/{config.n_games}: "
                  f"Wins A ({config.solver_a.type})={wins_a}, "
                  f"Wins B ({config.solver_b.type})={wins_b}, Draws={draws}")

    elapsed_s = time.time() - start_time

    summary = {
        "experiment_name": config.experiment_name,
        "n_games": config.n_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / config.n_games,
        "win_rate_b": wins_b / config.n_games,
        "draw_rate": draws / config.n_games,
        "total_duration_s": round(elapsed_s, 2),
    }

    summary_file = os.path.join(exp_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment Complete! Summary saved to: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Quoridor Experiment Runner Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON experiment config file")
    parser.add_argument("--solver-a", type=str, default="random", help="Solver A spec (e.g. 'mcts simulations=200')")
    parser.add_argument("--solver-b", type=str, default="dijkstra", help="Solver B spec (e.g. 'dijkstra')")
    parser.add_argument("--n-games", type=int, default=10, help="Number of games to execute")
    parser.add_argument("--seed", type=int, default=0, help="Random seed base")
    parser.add_argument("--name", type=str, default="experiment_run", help="Experiment name")
    parser.add_argument("--out-dir", type=str, default="experiments/results/", help="Results directory")
    parser.add_argument("--latency-budget-ms", type=float, default=None, help="Latency budget per move in ms")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable turn-level telemetry logging")
    parser.add_argument("--no-swap", action="store_true", help="Disable side swapping")

    args = parser.parse_args()

    if args.config:
        # Load config file (YAML or JSON)
        with open(args.config, "r") as f:
            if args.config.endswith(".json"):
                cfg_dict = json.load(f)
            else:
                try:
                    import yaml
                    cfg_dict = yaml.safe_load(f)
                except ImportError:
                    # Fallback JSON parsing if PyYAML not installed
                    cfg_dict = json.load(f)

        s_a = SolverSpec(**cfg_dict["solver_a"]) if isinstance(cfg_dict["solver_a"], dict) else parse_solver_spec_string(cfg_dict["solver_a"])
        s_b = SolverSpec(**cfg_dict["solver_b"]) if isinstance(cfg_dict["solver_b"], dict) else parse_solver_spec_string(cfg_dict["solver_b"])

        exp_cfg = ExperimentConfig(
            experiment_name=cfg_dict.get("experiment_name", "exp_from_config"),
            solver_a=s_a,
            solver_b=s_b,
            n_games=cfg_dict.get("n_games", 100),
            seed=cfg_dict.get("seed", 0),
            output_dir=cfg_dict.get("output_dir", "experiments/results/"),
            record_games=cfg_dict.get("record_games", True),
            record_telemetry=cfg_dict.get("record_telemetry", True),
            latency_budget_ms=cfg_dict.get("latency_budget_ms", None),
            swap_sides=cfg_dict.get("swap_sides", True),
        )
    else:
        s_a = parse_solver_spec_string(args.solver_a)
        s_b = parse_solver_spec_string(args.solver_b)

        exp_cfg = ExperimentConfig(
            experiment_name=args.name,
            solver_a=s_a,
            solver_b=s_b,
            n_games=args.n_games,
            seed=args.seed,
            output_dir=args.out_dir,
            record_telemetry=not args.no_telemetry,
            latency_budget_ms=args.latency_budget_ms,
            swap_sides=not args.no_swap,
        )

    run_experiment(exp_cfg)


if __name__ == "__main__":
    main()
