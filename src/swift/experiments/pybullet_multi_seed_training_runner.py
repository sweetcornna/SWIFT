from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import hashlib
import json

from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)
from swift.experiments.pybullet_training_runner import (
    PyBulletPPOTrainingRunConfig,
    run_pybullet_ppo_training,
)

if TYPE_CHECKING:
    from swift.config import SimulationSettings, TrainingSettings


@dataclass(frozen=True)
class PyBulletMultiSeedTrainingConfig:
    training_settings: TrainingSettings
    simulation_settings: SimulationSettings
    seeds: tuple[int, ...] = (8, 9, 10)
    total_timesteps: int | None = None
    output: Path | None = None

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("seeds must contain at least one seed")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if self.total_timesteps is not None and self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if not self.training_settings.pybullet_obstacle_randomization.enabled:
            raise ValueError("PyBullet obstacle randomization must be enabled")
        object.__setattr__(self, "seeds", seeds)
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))


def run_pybullet_multi_seed_training(config: PyBulletMultiSeedTrainingConfig) -> dict[str, Any]:
    total_timesteps = config.total_timesteps or config.training_settings.run.total_timesteps
    writer = ExperimentArtifactWriter(config.training_settings.artifact)
    config_hash = _config_hash(config, total_timesteps)
    stage = config.training_settings.run.stage
    variant = "pybullet_multi_seed_training"
    run_id = build_run_id(
        stage=stage,
        variant=variant,
        seed=min(config.seeds),
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(stage, variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json
    child_dir = output_path.parent / f"{output_path.stem}_seeds"

    child_reports = [
        run_pybullet_ppo_training(
            PyBulletPPOTrainingRunConfig(
                training_settings=config.training_settings,
                simulation_settings=config.simulation_settings,
                total_timesteps=total_timesteps,
                seed=seed,
                output=child_dir / f"seed_{seed}.json",
                enable_pybullet_obstacles=True,
            )
        )
        for seed in config.seeds
    ]
    seed_runs = [
        {
            "seed": seed,
            "run_id": str(report["run_id"]),
            "completed": _completed(report, total_timesteps),
            "metrics": dict(report["metrics"]),
            "training": dict(report["training"]),
            "artifacts": dict(report["artifacts"]),
        }
        for seed, report in zip(config.seeds, child_reports, strict=True)
    ]
    metrics = [run["metrics"] for run in seed_runs]
    report = {
        "schema_version": 1,
        "record_type": "pybullet_multi_seed_training_report",
        "stage": stage,
        "variant": variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lineage": {
            "config_hash": config_hash,
            "source_runner": "swift.experiments.pybullet_training_runner.run_pybullet_ppo_training",
            "training_backend": "torch_ppo_mlp_pybullet_velocity",
        },
        "seeds": list(config.seeds),
        "seed_count": len(config.seeds),
        "total_timesteps": total_timesteps,
        "aggregate_total_timesteps": total_timesteps * len(config.seeds),
        "metrics": {
            "worst_success_rate": min(float(item["success_rate"]) for item in metrics),
            "max_collision_rate": max(float(item["collision_rate"]) for item in metrics),
            "max_timeout_rate": max(float(item["timeout_rate"]) for item in metrics),
            "mean_average_episode_return": sum(float(item["average_episode_return"]) for item in metrics)
            / len(metrics),
        },
        "seed_runs": seed_runs,
        "readiness": {
            "all_seeds_completed": all(bool(run["completed"]) for run in seed_runs),
            "seed_count": len(config.seeds),
            "robustness_claim": False,
        },
        "artifacts": {
            "summary_json": str(output_path),
            "manifest_json": str(manifest_path),
        },
    }
    writer.write_summary(output_path, report)
    writer.write_manifest(
        manifest_path,
        build_artifact_manifest(
            subject_record_type=report["record_type"],
            run_id=run_id,
            stage=stage,
            variant=variant,
            lineage=report["lineage"],
            inputs=[
                artifact_reference(
                    run["artifacts"]["summary_json"],
                    role=f"seed_{run['seed']}_training_report",
                )
                for run in seed_runs
            ],
            outputs=[artifact_reference(output_path, role="multi_seed_training_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _completed(report: dict[str, Any], total_timesteps: int) -> bool:
    training = report.get("training", {})
    artifacts = report.get("artifacts", {})
    return (
        report.get("record_type") == "pybullet_ppo_training_report"
        and int(training.get("total_timesteps", 0)) >= total_timesteps
        and Path(str(artifacts.get("checkpoint_path", ""))).is_file()
    )


def _config_hash(config: PyBulletMultiSeedTrainingConfig, total_timesteps: int) -> str:
    material = repr(
        (
            config.training_settings,
            config.simulation_settings,
            config.seeds,
            total_timesteps,
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
