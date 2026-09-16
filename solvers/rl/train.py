"""
Config-Driven Training Module for Quoridor RL Agent.
Uses MaskablePPO from sb3-contrib with structured TrainConfig and run_summary.json export.
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn

from core.env import QuoridorEnv, SelfPlayEnv
from solvers.rl.feature_extractor import QuoridorCNN, QuoridorResidualCNN
from solvers.registry import get_agent


@dataclass
class TrainConfig:
    """Hyperparameters and configuration for PPO training run."""
    opponent: Optional[str] = "random"
    total_timesteps: int = 1_000_000
    feature_extractor: str = "cnn"       # "cnn" | "residual"
    heatmaps_enabled: bool = True         # RQ1 ablation flag
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 2048
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
    save: bool = True
    load_model: Optional[str] = None
    run_name: Optional[str] = None
    verbose: int = 1


def mask_fn(env):
    """Get action mask from the environment."""
    while hasattr(env, 'env'):
        env = env.env
    return env.action_masks()


def make_env(
    opponent_type: Optional[str] = None,
    heatmaps_enabled: bool = True,
    seed: int = 0,
    rank: int = 0,
):
    """Factory helper to create wrapped environments."""
    def _init():
        current_seed = None if seed is None else seed + rank
        if opponent_type:
            opponent = get_agent(opponent_type, player=2, seed=current_seed)
            env = QuoridorEnv(
                opponent=opponent,
                heatmaps_enabled=heatmaps_enabled,
                max_steps=200,
            )
        else:
            env = SelfPlayEnv(max_steps=200)

        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def train(config: TrainConfig) -> Dict[str, Any]:
    """
    Programmatic training entry point using TrainConfig dataclass.
    Returns summary dict containing run stats & artifact paths.
    """
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.run_name or f"quoridor_ppo_{config.opponent or 'selfplay'}_{timestamp}"

    start_time = time.time()

    # Create vectorized environments
    if config.n_envs > 1:
        env = SubprocVecEnv([
            make_env(config.opponent, config.heatmaps_enabled, config.seed, i)
            for i in range(config.n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(config.opponent, config.heatmaps_enabled, config.seed, 0)
        ])

    # Select feature extractor
    if config.feature_extractor == "residual":
        extractor_class = QuoridorResidualCNN
    else:
        extractor_class = QuoridorCNN

    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
    }

    if config.load_model:
        if config.verbose > 0:
            print(f"Loading model from: {config.load_model}")
        model = MaskablePPO.load(
            config.load_model,
            env=env,
            tensorboard_log=config.log_dir,
            verbose=config.verbose,
        )
        model.lr_schedule = get_schedule_fn(config.learning_rate)
        model.learning_rate = config.learning_rate
        if model.policy.optimizer is not None:
            for pg in model.policy.optimizer.param_groups:
                pg['lr'] = config.learning_rate
    else:
        model = MaskablePPO(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=config.log_dir,
            verbose=config.verbose,
            seed=config.seed,
        )

    callbacks = []
    if config.save:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(config.save_freq // config.n_envs, 1),
            save_path=config.model_dir,
            name_prefix=f"quoridor_{config.opponent or 'selfplay'}",
        )
        callbacks.append(checkpoint_callback)

    if config.verbose > 0:
        print(f"\n{'='*60}")
        print(f"Training Quoridor PPO Agent")
        print(f"{'='*60}")
        print(f"Run Name: {run_name}")
        print(f"Opponent: {config.opponent or 'self-play'}")
        print(f"Heatmaps Enabled: {config.heatmaps_enabled}")
        print(f"Total Timesteps: {config.total_timesteps:,}")
        print(f"{'='*60}\n")

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
        tb_log_name=run_name,
    )

    elapsed_s = time.time() - start_time
    final_path = None

    if config.save:
        final_path = os.path.join(config.model_dir, f"{run_name}_final.zip")
        model.save(final_path)

        # Save structured run summary JSON
        summary = {
            "run_name": run_name,
            "timestamp": timestamp,
            "training_duration_s": round(elapsed_s, 2),
            "final_model_path": final_path,
            "config": asdict(config),
        }
        summary_path = os.path.join(config.model_dir, f"{run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if config.verbose > 0:
            print(f"\nModel saved to: {final_path}")
            print(f"Run summary saved to: {summary_path}")

    env.close()

    return {
        "model": model,
        "run_name": run_name,
        "final_model_path": final_path,
        "elapsed_s": elapsed_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Quoridor RL Agent")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--opponent", type=str, default="random")
    parser.add_argument("--extractor", type=str, default="cnn")
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    cfg = TrainConfig(
        total_timesteps=args.timesteps,
        opponent=args.opponent,
        feature_extractor=args.extractor,
        heatmaps_enabled=not args.no_heatmaps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        save=not args.no_save,
        load_model=args.load_model,
        verbose=args.verbose,
    )
    train(cfg)


if __name__ == "__main__":
    main()
