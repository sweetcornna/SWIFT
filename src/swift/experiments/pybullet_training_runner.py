from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import hashlib
import json

from swift.envs import PyBulletVelocityTrainingEnv
from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
    checkpoint_filename,
)
from swift.rl.ppo import MLPActorCriticConfig, PPOTrainingConfig, PPOTrainingResult, train_ppo_mlp

if TYPE_CHECKING:
    from swift.config import SimulationSettings, TrainingSettings


@dataclass(frozen=True)
class PyBulletPPOTrainingRunConfig:
    training_settings: TrainingSettings
    simulation_settings: SimulationSettings
    total_timesteps: int | None = None
    seed: int | None = None
    output: Path | None = None
    enable_pybullet_obstacles: bool = False
    velocity_aviary_cls: Any | None = None
    drone_model: Any | None = None
    physics: Any | None = None

    def __post_init__(self) -> None:
        if self.total_timesteps is not None and self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))


def run_pybullet_ppo_training(config: PyBulletPPOTrainingRunConfig) -> dict[str, Any]:
    base_training_config = _training_config(config)
    writer = ExperimentArtifactWriter(config.training_settings.artifact)
    config_hash = _config_hash(
        config.training_settings,
        config.simulation_settings,
        base_training_config.total_timesteps,
        config.enable_pybullet_obstacles,
    )
    run_id = build_run_id(
        stage=config.training_settings.run.stage,
        variant=config.training_settings.run.variant,
        seed=base_training_config.seed,
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(
        config.training_settings.run.stage,
        config.training_settings.run.variant,
        run_id,
    )
    output_path = config.output or paths.summary_json
    manifest_path = _manifest_sidecar_path(output_path) if config.output is not None else paths.manifest_json
    checkpoint_path = paths.checkpoint_dir / checkpoint_filename(
        variant=config.training_settings.run.variant,
        run_id=run_id,
        episode=0,
        step=base_training_config.total_timesteps,
        label="final",
        metric_name="planned_steps",
        metric_value=float(base_training_config.total_timesteps),
    )
    training_config = _training_config(
        config,
        checkpoint_path=checkpoint_path,
        history_path=paths.episode_jsonl,
    )

    result = train_ppo_mlp(
        lambda: PyBulletVelocityTrainingEnv(
            simulation_settings=config.simulation_settings,
            settings=config.training_settings.environment,
            enable_pybullet_obstacles=config.enable_pybullet_obstacles,
            obstacle_randomization=config.training_settings.pybullet_obstacle_randomization,
            reward_settings=config.training_settings.pybullet_reward,
            curriculum=config.training_settings.pybullet_curriculum,
            velocity_aviary_cls=config.velocity_aviary_cls,
            drone_model=config.drone_model,
            physics=config.physics,
        ),
        training_config,
    )
    summary = _summary_payload(
        config=config,
        result=result,
        run_id=run_id,
        output_path=output_path,
        history_path=paths.episode_jsonl,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        config_hash=config_hash,
    )
    writer.write_summary(output_path, summary)
    writer.write_manifest(
        manifest_path,
        build_artifact_manifest(
            subject_record_type=summary["record_type"],
            run_id=run_id,
            stage=config.training_settings.run.stage,
            variant=config.training_settings.run.variant,
            lineage=summary["lineage"],
            outputs=[
                artifact_reference(output_path, role="summary_json"),
                artifact_reference(paths.episode_jsonl, role="training_history_jsonl"),
                artifact_reference(checkpoint_path, role="checkpoint"),
            ],
        ),
    )
    return json.loads(Path(output_path).read_text(encoding="utf-8"))


def _training_config(
    config: PyBulletPPOTrainingRunConfig,
    *,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
) -> PPOTrainingConfig:
    total_timesteps = config.total_timesteps or config.training_settings.run.total_timesteps
    rollout_steps = min(config.training_settings.ppo.rollout_steps, total_timesteps)
    minibatch_size = min(config.training_settings.ppo.minibatch_size, rollout_steps)
    update_epochs = config.training_settings.ppo.update_epochs
    return PPOTrainingConfig(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        minibatch_size=minibatch_size,
        update_epochs=update_epochs,
        clip_range=config.training_settings.ppo.clip_range,
        gamma=config.training_settings.ppo.gamma,
        gae_lambda=config.training_settings.ppo.gae_lambda,
        entropy_coef=config.training_settings.ppo.entropy_coef,
        value_loss_coef=config.training_settings.ppo.value_loss_coef,
        max_grad_norm=config.training_settings.ppo.max_grad_norm,
        seed=config.seed if config.seed is not None else config.training_settings.run.seed,
        torch_num_threads=1,
        network=MLPActorCriticConfig(
            max_heading_delta=config.training_settings.policy.max_heading_delta,
            min_speed_fraction=config.training_settings.policy.min_speed_fraction,
        ),
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


def _summary_payload(
    *,
    config: PyBulletPPOTrainingRunConfig,
    result: PPOTrainingResult,
    run_id: str,
    output_path: Path,
    history_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    config_hash: str,
) -> dict[str, Any]:
    training = asdict(result)
    return {
        "schema_version": 1,
        "record_type": "pybullet_ppo_training_report",
        "stage": config.training_settings.run.stage,
        "variant": config.training_settings.run.variant,
        "run_id": run_id,
        "lineage": {
            "training_backend": "torch_ppo_mlp_pybullet_velocity",
            "environment_adapter": "swift.envs.PyBulletVelocityTrainingEnv",
            "config_hash": config_hash,
        },
        "runtime": {
            "runtime_contract": "pybullet_velocity_training_compatibility",
            "pybullet_root": str(config.simulation_settings.pybullet_root),
            "pixi_executable": str(config.simulation_settings.pixi_executable),
            "enable_pybullet_obstacles": bool(config.enable_pybullet_obstacles),
            "obstacle_randomization": asdict(config.training_settings.pybullet_obstacle_randomization),
            "reward": asdict(config.training_settings.pybullet_reward),
            "curriculum": asdict(config.training_settings.pybullet_curriculum),
        },
        "training": training,
        "metrics": {
            "success_rate": training["success_rate"],
            "collision_rate": training["collision_rate"],
            "timeout_rate": training["timeout_rate"],
            "average_episode_return": training["average_episode_return"],
        },
        "artifacts": {
            "summary_json": str(output_path),
            "training_history_jsonl": str(history_path),
            "checkpoint_path": str(checkpoint_path),
            "manifest_json": str(manifest_path),
        },
    }


def _config_hash(
    training_settings: TrainingSettings,
    simulation_settings: SimulationSettings,
    total_timesteps: int,
    enable_pybullet_obstacles: bool,
) -> str:
    material = repr(
        (
            training_settings,
            simulation_settings,
            total_timesteps,
            bool(enable_pybullet_obstacles),
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _manifest_sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")
