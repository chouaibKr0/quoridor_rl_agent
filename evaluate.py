"""
Evaluation Script for Quoridor RL Agent
Tests trained models against various opponent agents
"""

import argparse
import os
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from quoridor_env import QuoridorEnv
from ai_agent import get_agent, BaseAgent

DETERMINISTIC = False

def evaluate(
    model_path: str,
    opponent: str = "random",
    n_episodes: int = 100,
    render: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """
    Evaluate a trained model against an opponent.
    
    Args:
        model_path: Path to the saved model
        opponent: Opponent type ('random', 'dijkstra', 'strategic', 'minimax')
        n_episodes: Number of evaluation episodes
        render: Whether to render the game
        seed: Random seed
    
    Returns:
        Dictionary with evaluation statistics
    """
    # Load model
    model = MaskablePPO.load(model_path)
    
    # Create opponent
    opponent_agent = get_agent(opponent, player=2, seed=seed)
    
    # Create environment
    env = QuoridorEnv(
        opponent=opponent_agent,
        render_mode="human" if render else None,
        max_steps=200,
    )
    
    # Statistics
    wins = 0
    losses = 0
    draws = 0
    episode_lengths = []
    episode_rewards = []
    
    print(f"\nEvaluating against {opponent} agent ({n_episodes} episodes)...")
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode if seed else None)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Get action from model with action masking
            action_mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=DETERMINISTIC)

            # Extract scalar if it's an array
            if isinstance(action, np.ndarray):
                action = action.item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_lengths.append(steps)
        episode_rewards.append(episode_reward)
        
        # Determine outcome
        if terminated:
            if env.game.p1_pos[0] == 8:  # P1 (agent) won
                wins += 1
            elif env.game.p2_pos[0] == 0:  # P2 (opponent) won
                losses += 1
            else:
                draws += 1
        else:
            draws += 1  # Truncated = draw
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: "
                  f"Wins={wins}, Losses={losses}, Draws={draws}")
    
    env.close()
    
    # Calculate statistics
    results = {
        "opponent": opponent,
        "n_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
    }
    
    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results vs {results['opponent'].upper()}")
    print(f"{'='*60}")
    print(f"Episodes: {results['n_episodes']}")
    print(f"Wins: {results['wins']} ({results['win_rate']*100:.1f}%)")
    print(f"Losses: {results['losses']} ({results['loss_rate']*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draw_rate']*100:.1f}%)")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
    print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"{'='*60}\n")


def evaluate_all(model_path: str, n_episodes: int = 100, seed: Optional[int] = None):
    """Evaluate model against all opponent types."""
    opponents = ["random", "dijkstra", "strategic"]
    all_results = []
    
    for opponent in opponents:
        results = evaluate(model_path, opponent, n_episodes, render=False, seed=seed)
        print_results(results)
        all_results.append(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Opponent':<15} {'Win Rate':>10} {'Loss Rate':>10} {'Draw Rate':>10}")
    print(f"{'-'*45}")
    for r in all_results:
        print(f"{r['opponent']:<15} {r['win_rate']*100:>9.1f}% {r['loss_rate']*100:>9.1f}% {r['draw_rate']*100:>9.1f}%")
    print(f"{'='*60}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quoridor RL Agent")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model (.zip)")
    parser.add_argument("--opponent", type=str, default=None,
                        choices=["random", "dijkstra", "strategic", "minimax", "all"],
                        help="Opponent type (or 'all' for all opponents)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the game")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if args.opponent == "all":
        evaluate_all(args.model, args.episodes, args.seed)
    else:
        opponent = args.opponent or "random"
        results = evaluate(args.model, opponent, args.episodes, args.render, args.seed)
        print_results(results)


if __name__ == "__main__":
    main()
