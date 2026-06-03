from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json

from swift.config import TrainingSettings
from swift.experiments.artifacts import ExperimentArtifactWriter, build_run_id
from swift.rl.ppo import PPOTrainingConfig, PPOTrainingResult, train_ppo_mlp
from swift.envs import SimpleAvoidanceEnv


@dataclass(frozen=True)
class PPOTrainingRunConfig:
    settings: TrainingSettings
    total_timesteps: int | None = None
    seed: int | None = None
    output: Path | None = None

    def __post_init__(self) -> None:
        if self.total_timesteps is not None and self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))


def run_ppo_training_smoke(config: PPOTrainingRunConfig) -> dict[str, Any]:
    training_config = _training_config(config)
    writer = ExperimentArtifactWriter(config.settings.artifact)
    run_id = build_run_id(
        stage=config.settings.run.stage,
        variant=config.settings.run.variant,
        seed=training_config.seed,
        config_hash=_config_hash(config.settings, training_config.total_timesteps),
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(config.settings.run.stage, config.settings.run.variant, run_id)
    output_path = config.output or paths.summary_json

    result = train_ppo_mlp(
        lambda: SimpleAvoidanceEnv(config.settings.environment),
        training_config,
    )
    summary = _summary_payload(
        settings=config.settings,
        result=result,
        run_id=run_id,
        output_path=output_path,
    )
    writer.write_summary(output_path, summary)
    return json.loads(Path(output_path).read_text(encoding="utf-8"))


def _training_config(config: PPOTrainingRunConfig) -> PPOTrainingConfig:
    total_timesteps = config.total_timesteps or config.settings.run.total_timesteps
    rollout_steps = min(config.settings.ppo.rollout_steps, total_timesteps, 64)
    minibatch_size = min(config.settings.ppo.minibatch_size, rollout_steps)
    update_epochs = min(config.settings.ppo.update_epochs, 2)
    return PPOTrainingConfig(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        minibatch_size=minibatch_size,
        update_epochs=update_epochs,
        clip_range=config.settings.ppo.clip_range,
        gamma=config.settings.ppo.gamma,
        gae_lambda=config.settings.ppo.gae_lambda,
        entropy_coef=config.settings.ppo.entropy_coef,
        value_loss_coef=config.settings.ppo.value_loss_coef,
        max_grad_norm=config.settings.ppo.max_grad_norm,
        seed=config.seed if config.seed is not None else config.settings.run.seed,
        torch_num_threads=1,
    )


def _summary_payload(
    *,
    settings: TrainingSettings,
    result: PPOTrainingResult,
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    training = asdict(result)
    return {
        "schema_version": 1,
        "record_type": "ppo_training_smoke",
        "stage": settings.run.stage,
        "variant": settings.run.variant,
        "run_id": run_id,
        "training": training,
        "metrics": {
            "success_rate": training["success_rate"],
            "collision_rate": training["collision_rate"],
            "timeout_rate": training["timeout_rate"],
            "average_episode_return": training["average_episode_return"],
        },
        "artifacts": {
            "summary_json": str(output_path),
        },
    }


def _config_hash(settings: TrainingSettings, total_timesteps: int) -> str:
    material = repr((settings, total_timesteps)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
