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
from swift.experiments.pybullet_checkpoint_evaluator import (
    PyBulletCheckpointEvaluationConfig,
    run_pybullet_checkpoint_evaluation,
)

if TYPE_CHECKING:
    from swift.config import SimulationSettings, TrainingSettings


@dataclass(frozen=True)
class PyBulletMultiSeedCheckpointEvaluationConfig:
    training_report_path: Path
    training_settings: TrainingSettings
    simulation_settings: SimulationSettings
    training_config_path: Path | None = None
    simulation_config_path: Path | None = None
    episodes_per_checkpoint: int = 100
    holdout_seed: int = 1_000_000
    output: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_report_path", Path(self.training_report_path))
        if self.training_config_path is not None:
            object.__setattr__(self, "training_config_path", Path(self.training_config_path))
        if self.simulation_config_path is not None:
            object.__setattr__(self, "simulation_config_path", Path(self.simulation_config_path))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        if self.episodes_per_checkpoint <= 0:
            raise ValueError("episodes_per_checkpoint must be positive")
        if self.holdout_seed < 0:
            raise ValueError("holdout_seed must be non-negative")
        if not self.training_settings.pybullet_obstacle_randomization.enabled:
            raise ValueError("PyBullet obstacle randomization must be enabled")


def run_pybullet_multi_seed_checkpoint_evaluation(
    config: PyBulletMultiSeedCheckpointEvaluationConfig,
) -> dict[str, Any]:
    source = _load_training_report(config.training_report_path)
    seed_runs = source.get("seed_runs", [])
    if not isinstance(seed_runs, list) or not seed_runs:
        raise ValueError("training report must contain seed_runs")
    training_seeds = {int(run["seed"]) for run in seed_runs}
    holdout_seeds = set(range(config.holdout_seed, config.holdout_seed + config.episodes_per_checkpoint))
    if training_seeds & holdout_seeds:
        raise ValueError("holdout seeds must not overlap training seeds")

    writer = ExperimentArtifactWriter(config.training_settings.artifact)
    config_hash = _config_hash(config, source)
    stage = config.training_settings.run.stage
    variant = "pybullet_multi_seed_checkpoint_holdout"
    run_id = build_run_id(
        stage=stage,
        variant=variant,
        seed=config.holdout_seed,
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(stage, variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json
    child_dir = output_path.parent / f"{output_path.stem}_checkpoint_evals"

    evaluations = []
    for seed_run in seed_runs:
        training_seed = int(seed_run["seed"])
        checkpoint_path = Path(str(seed_run.get("artifacts", {}).get("checkpoint_path", "")))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))
        evaluation = run_pybullet_checkpoint_evaluation(
            PyBulletCheckpointEvaluationConfig(
                checkpoint_path=checkpoint_path,
                training_settings=config.training_settings,
                simulation_settings=config.simulation_settings,
                training_config_path=config.training_config_path,
                simulation_config_path=config.simulation_config_path,
                output=child_dir / f"train_seed_{training_seed}.json",
                episodes=config.episodes_per_checkpoint,
                seed=config.holdout_seed,
                enable_pybullet_obstacles=True,
            )
        )
        evaluations.append(
            {
                "training_seed": training_seed,
                "run_id": str(evaluation["run_id"]),
                "completed": (
                    evaluation.get("record_type") == "pybullet_ppo_checkpoint_evaluation"
                    and int(evaluation.get("episodes_requested", 0)) == config.episodes_per_checkpoint
                ),
                "checkpoint_path": str(checkpoint_path),
                "metrics": dict(evaluation["metrics"]),
                "artifacts": dict(evaluation["artifacts"]),
            }
        )

    metrics = [evaluation["metrics"] for evaluation in evaluations]
    min_total_timesteps = min(int(run.get("training", {}).get("total_timesteps", 0)) for run in seed_runs)
    report = {
        "schema_version": 1,
        "record_type": "pybullet_multi_seed_checkpoint_holdout_report",
        "stage": stage,
        "variant": variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run_id": str(source.get("run_id", "")),
        "lineage": {
            "config_hash": config_hash,
            "source_runner": "swift.experiments.pybullet_checkpoint_evaluator.run_pybullet_checkpoint_evaluation",
            "evaluation_backend": "deterministic_randomized_pybullet_checkpoint_holdout",
        },
        "training": {
            "seed_count": len(seed_runs),
            "seeds": sorted(training_seeds),
            "min_total_timesteps": min_total_timesteps,
            "aggregate_total_timesteps": sum(
                int(run.get("training", {}).get("total_timesteps", 0)) for run in seed_runs
            ),
        },
        "holdout": {
            "seed_start": config.holdout_seed,
            "seed_end": config.holdout_seed + config.episodes_per_checkpoint - 1,
            "episodes_per_checkpoint": config.episodes_per_checkpoint,
            "total_episodes": config.episodes_per_checkpoint * len(evaluations),
        },
        "metrics": {
            "worst_success_rate": min(float(item["success_rate"]) for item in metrics),
            "max_collision_rate": max(float(item["collision_rate"]) for item in metrics),
            "max_timeout_rate": max(float(item["timeout_rate"]) for item in metrics),
            "worst_average_minimum_safety_distance": min(
                float(item["average_minimum_safety_distance"]) for item in metrics
            ),
        },
        "checkpoint_evaluations": evaluations,
        "readiness": {
            "all_checkpoints_evaluated": all(bool(item["completed"]) for item in evaluations),
            "training_seed_count": len(seed_runs),
            "robustness_claim": False,
        },
        "artifacts": {
            "summary_json": str(output_path),
            "manifest_json": str(manifest_path),
            "training_report_json": str(config.training_report_path),
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
                artifact_reference(config.training_report_path, role="multi_seed_training_report"),
                *[
                    artifact_reference(
                        item["artifacts"]["summary_json"],
                        role=f"train_seed_{item['training_seed']}_checkpoint_evaluation",
                    )
                    for item in evaluations
                ],
            ],
            outputs=[artifact_reference(output_path, role="multi_seed_checkpoint_holdout_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _load_training_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("record_type") != "pybullet_multi_seed_training_report":
        raise ValueError("input must be a pybullet_multi_seed_training_report")
    if not report.get("readiness", {}).get("all_seeds_completed", False):
        raise ValueError("all training seeds must be completed before holdout evaluation")
    return report


def _config_hash(
    config: PyBulletMultiSeedCheckpointEvaluationConfig,
    source: dict[str, Any],
) -> str:
    material = repr(
        (
            source.get("run_id"),
            config.training_settings,
            config.simulation_settings,
            config.episodes_per_checkpoint,
            config.holdout_seed,
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
