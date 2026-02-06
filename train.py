"""
Training Script for Quoridor RL Agent
Uses MaskablePPO from sb3-contrib with custom CNN feature extractor
"""

import argparse
import os
from datetime import datetime
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from quoridor_env import QuoridorEnv, SelfPlayEnv
from feature_extractor import QuoridorCNN, QuoridorResidualCNN
from ai_agent import get_agent


def mask_fn(env):
    """Get action mask from the environment."""
    # Unwrap Monitor wrapper to access base environment
    while hasattr(env, 'env'):
        env = env.env
    return env.action_masks()


def make_env(opponent_type: Optional[str] = None, seed: int = 0, rank: int = 0):
    """
    Create a wrapped Quoridor environment.
    
    Args:
        opponent_type: Type of opponent ('random', 'dijkstra', 'strategic', None for self-play)
        seed: Random seed
        rank: Environment rank for vectorized envs
    """
    def _init():
        if opponent_type:
            opponent = get_agent(opponent_type, player=2, seed=seed + rank)
            env = QuoridorEnv(opponent=opponent, max_steps=200)
        else:
            env = SelfPlayEnv(max_steps=200)
        
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    
    return _init


def train(
    total_timesteps: int = 1_000_000,
    opponent: Optional[str] = None,
    feature_extractor: str = "cnn",
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    save_freq: int = 50_000,
    log_dir: str = "./logs",
    model_dir: str = "./models",
    seed: int = 42,
    save: bool = True,
    verbose: int = 1,
):
    """
    Train a Quoridor agent using MaskablePPO.
    
    Args:
        total_timesteps: Total training steps
        opponent: Opponent type for training ('random', 'dijkstra', 'strategic', None for self-play)
        feature_extractor: CNN architecture ('cnn' or 'residual')
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        save_freq: Save checkpoint every N steps
        log_dir: TensorBoard log directory
        model_dir: Model save directory
        seed: Random seed
        save: Whether to save the model
        verbose: Verbosity level
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Timestamp for unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"quoridor_ppo_{opponent or 'selfplay'}_{timestamp}"
    
    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(opponent, seed, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(opponent, seed, 0)])
    
    # Select feature extractor
    if feature_extractor == "residual":
        extractor_class = QuoridorResidualCNN
    else:
        extractor_class = QuoridorCNN
    
    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
    }
    
    # Create model
    model = MaskablePPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=verbose,
        seed=seed,
    )
    
    # Callbacks
    callbacks = []
    
    if save:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(save_freq // n_envs, 1),
            save_path=model_dir,
            name_prefix=f"quoridor_{opponent or 'selfplay'}",
        )
        callbacks.append(checkpoint_callback)
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training Quoridor PPO Agent")
    print(f"{'='*60}")
    print(f"Opponent: {opponent or 'self-play'}")
    print(f"Feature Extractor: {feature_extractor}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Envs: {n_envs}")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
        tb_log_name=run_name,
    )
    
    # Save final model
    if save:
        final_path = os.path.join(model_dir, f"quoridor_ppo_final_{timestamp}.zip")
        model.save(final_path)
        print(f"\nModel saved to: {final_path}")
    
    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Quoridor RL Agent")
    
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--opponent", type=str, default=None,
                        choices=["random", "dijkstra", "strategic", None],
                        help="Opponent type (None for self-play)")
    parser.add_argument("--extractor", type=str, default="cnn",
                        choices=["cnn", "residual"],
                        help="Feature extractor architecture")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Checkpoint save frequency")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="TensorBoard log directory")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Model save directory")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save the model")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        opponent=args.opponent,
        feature_extractor=args.extractor,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        save=not args.no_save,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
