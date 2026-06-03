from __future__ import annotations

from dataclasses import dataclass
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
)
from swift.experiments.ppo_checkpoint_evaluator import (
    _aggregate_metrics,
    _assert_json_safe,
    _load_checkpoint_policy,
)

if TYPE_CHECKING:
    from swift.config import SimulationSettings, TrainingSettings


@dataclass(frozen=True)
class PyBulletCheckpointEvaluationConfig:
    checkpoint_path: Path
    training_settings: TrainingSettings
    simulation_settings: SimulationSettings
    training_config_path: Path | None = None
    simulation_config_path: Path | None = None
    output: Path | None = None
    episodes: int = 3
    seed: int = 0
    enable_pybullet_obstacles: bool = False
    velocity_aviary_cls: Any | None = None
    drone_model: Any | None = None
    physics: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.training_config_path is not None:
            object.__setattr__(self, "training_config_path", Path(self.training_config_path))
        if self.simulation_config_path is not None:
            object.__setattr__(self, "simulation_config_path", Path(self.simulation_config_path))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def run_pybullet_checkpoint_evaluation(config: PyBulletCheckpointEvaluationConfig) -> dict[str, Any]:
    model, checkpoint, deterministic_action, policy_family = _load_checkpoint_policy(config.checkpoint_path)
    if checkpoint.get("record_type") != "ppo_checkpoint":
        raise ValueError("PyBullet checkpoint evaluation currently supports ppo_checkpoint only")
    writer = ExperimentArtifactWriter(config.training_settings.artifact)
    config_hash = _config_hash(config)
    run_id = build_run_id(
        stage=config.training_settings.run.stage,
        variant=f"{config.training_settings.run.variant}_pybullet_eval",
        seed=config.seed,
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(
        config.training_settings.run.stage,
        f"{config.training_settings.run.variant}_pybullet_eval",
        run_id,
    )
    output_path = config.output or paths.summary_json
    manifest_path = _manifest_sidecar_path(output_path) if config.output is not None else paths.manifest_json
    episodes = [
        _evaluate_pybullet_episode(
            config=config,
            model=model,
            deterministic_action=deterministic_action,
            seed=config.seed + index,
        )
        for index in range(config.episodes)
    ]
    summary = {
        "schema_version": 1,
        "record_type": "pybullet_ppo_checkpoint_evaluation",
        "stage": config.training_settings.run.stage,
        "variant": f"{config.training_settings.run.variant}_pybullet_eval",
        "run_id": run_id,
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_record_type": str(checkpoint.get("record_type", "")),
        "policy_family": policy_family,
        "lineage": {
            "evaluation_backend": "deterministic_pybullet_checkpoint_eval",
            "environment_adapter": "swift.envs.PyBulletVelocityTrainingEnv",
            "config_hash": config_hash,
        },
        "runtime": {
            "runtime_contract": "pybullet_velocity_training_compatibility",
            "pybullet_root": str(config.simulation_settings.pybullet_root),
            "pixi_executable": str(config.simulation_settings.pixi_executable),
            "enable_pybullet_obstacles": bool(config.enable_pybullet_obstacles),
        },
        "checkpoint_training": dict(checkpoint.get("result", {})),
        "episodes_requested": config.episodes,
        "seed": config.seed,
        "metrics": _aggregate_metrics(episodes),
        "episodes": episodes,
        "artifacts": {
            "summary_json": str(output_path),
            "manifest_json": str(manifest_path),
            "checkpoint_path": str(config.checkpoint_path),
        },
    }
    if config.training_config_path is not None:
        summary["artifacts"]["training_config_yaml"] = str(config.training_config_path)
    if config.simulation_config_path is not None:
        summary["artifacts"]["simulation_config_yaml"] = str(config.simulation_config_path)
    _assert_json_safe(summary)
    writer.write_summary(output_path, summary)
    inputs = [artifact_reference(config.checkpoint_path, role="checkpoint")]
    if config.training_config_path is not None and config.training_config_path.is_file():
        inputs.append(artifact_reference(config.training_config_path, role="training_config_yaml"))
    if config.simulation_config_path is not None and config.simulation_config_path.is_file():
        inputs.append(artifact_reference(config.simulation_config_path, role="simulation_config_yaml"))
    writer.write_manifest(
        manifest_path,
        build_artifact_manifest(
            subject_record_type=summary["record_type"],
            run_id=run_id,
            stage=config.training_settings.run.stage,
            variant=f"{config.training_settings.run.variant}_pybullet_eval",
            lineage=summary["lineage"],
            inputs=inputs,
            outputs=[artifact_reference(output_path, role="summary_json")],
        ),
    )
    return json.loads(Path(output_path).read_text(encoding="utf-8"))


def _evaluate_pybullet_episode(
    *,
    config: PyBulletCheckpointEvaluationConfig,
    model: Any,
    deterministic_action: Any,
    seed: int,
) -> dict[str, Any]:
    env = PyBulletVelocityTrainingEnv(
        simulation_settings=config.simulation_settings,
        settings=config.training_settings.environment,
        enable_pybullet_obstacles=config.enable_pybullet_obstacles,
        velocity_aviary_cls=config.velocity_aviary_cls,
        drone_model=config.drone_model,
        physics=config.physics,
    )
    try:
        observation, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        info: dict[str, Any] = {}
        while not terminated and not truncated:
            action = deterministic_action(model, observation, env.settings, model.config)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
    finally:
        env.close()

    metrics = info["episode_metrics"]
    return {
        "seed": seed,
        "success": bool(metrics.success),
        "collided": bool(metrics.collided),
        "timed_out": bool(metrics.timed_out),
        "steps": int(metrics.steps),
        "return": total_reward,
        "path_length": float(metrics.path_length),
        "path_smoothness": float(metrics.path_smoothness),
        "minimum_safety_distance": float(metrics.minimum_safety_distance),
    }


def _config_hash(config: PyBulletCheckpointEvaluationConfig) -> str:
    material = repr(
        (
            config.checkpoint_path,
            config.training_settings,
            config.simulation_settings,
            config.episodes,
            config.seed,
            bool(config.enable_pybullet_obstacles),
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _manifest_sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")
