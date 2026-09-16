"""
Automated Curriculum Training Pipeline.
Executes sequential training stages against increasingly complex opponents
with win-rate threshold verification and adaptive extension logic.
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from solvers.rl.train import train, TrainConfig
from evaluate import evaluate


@dataclass
class CurriculumStage:
    """Definition of a single stage in the curriculum."""
    opponent: Optional[str]                      # Opponent type ("random", "dijkstra", "strategic", None)
    timesteps: int                               # Base timesteps for this stage
    win_rate_threshold: Optional[float] = None  # Required win rate to advance (e.g. 0.80)
    eval_episodes: int = 50                      # Number of episodes for win rate evaluation
    load_previous: bool = True                   # Continue training from previous model
    max_extensions: int = 2                      # Max times to extend training by 50% if threshold not met


class CurriculumRunner:
    """
    Executes a multi-stage curriculum pipeline.
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        base_config: Optional[TrainConfig] = None,
        model_dir: str = "./models/curriculum",
        log_dir: str = "./logs/curriculum",
    ):
        self.stages = stages
        self.base_config = base_config or TrainConfig()
        self.model_dir = model_dir
        self.log_dir = log_dir

    def run(self) -> Dict[str, Any]:
        """Run all curriculum stages sequentially."""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        start_time = time.time()
        stage_history = []
        last_model_path = self.base_config.load_model

        for idx, stage in enumerate(self.stages):
            stage_name = f"stage_{idx + 1}_{stage.opponent or 'selfplay'}"
            print(f"\n{'='*70}")
            print(f"STARTING CURRICULUM STAGE {idx + 1}/{len(self.stages)}: {stage_name}")
            print(f"{'='*70}")

            current_timesteps = stage.timesteps
            extensions_count = 0
            stage_passed = False
            last_win_rate = 0.0

            while not stage_passed and extensions_count <= stage.max_extensions:
                run_name = f"{stage_name}_ext{extensions_count}" if extensions_count > 0 else stage_name
                
                # Build TrainConfig for this stage attempt
                cfg = TrainConfig(
                    opponent=stage.opponent,
                    total_timesteps=current_timesteps,
                    feature_extractor=self.base_config.feature_extractor,
                    heatmaps_enabled=self.base_config.heatmaps_enabled,
                    n_envs=self.base_config.n_envs,
                    learning_rate=self.base_config.learning_rate,
                    batch_size=self.base_config.batch_size,
                    seed=self.base_config.seed,
                    save_freq=self.base_config.save_freq,
                    log_dir=self.log_dir,
                    model_dir=self.model_dir,
                    load_model=last_model_path if stage.load_previous else None,
                    run_name=run_name,
                    save=True,
                )

                # Train
                res = train(cfg)
                last_model_path = res["final_model_path"]

                # Evaluate win rate if opponent & threshold specified
                if stage.opponent and stage.win_rate_threshold is not None:
                    eval_res = evaluate(
                        model_path=last_model_path,
                        opponent=stage.opponent,
                        n_episodes=stage.eval_episodes,
                        seed=self.base_config.seed,
                    )
                    last_win_rate = eval_res["win_rate"]
                    print(f"Stage {idx + 1} Win Rate vs {stage.opponent}: {last_win_rate*100:.1f}% "
                          f"(Threshold: {stage.win_rate_threshold*100:.1f}%)")

                    if last_win_rate >= stage.win_rate_threshold:
                        stage_passed = True
                        print(f"✅ Stage {idx + 1} Threshold MET!")
                    else:
                        extensions_count += 1
                        if extensions_count <= stage.max_extensions:
                            current_timesteps = int(stage.timesteps * 0.5)
                            print(f"⚠️ Threshold not met. Extending training by {current_timesteps:,} steps...")
                else:
                    stage_passed = True
                    last_win_rate = 1.0

            stage_history.append({
                "stage": idx + 1,
                "opponent": stage.opponent,
                "timesteps_requested": stage.timesteps,
                "extensions_count": extensions_count,
                "final_win_rate": last_win_rate,
                "model_path": last_model_path,
                "passed": stage_passed,
            })

        summary = {
            "total_duration_s": round(time.time() - start_time, 2),
            "final_model_path": last_model_path,
            "stages": stage_history,
        }

        summary_path = os.path.join(self.model_dir, "curriculum_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nCurriculum Pipeline Complete! Summary saved to: {summary_path}")
        return summary
