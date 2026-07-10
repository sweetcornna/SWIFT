from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import hashlib
import json
import math

from swift.core import DroneAction
from swift.envs import PyBulletVelocityTrainingEnv
from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)

if TYPE_CHECKING:
    from swift.config import SimulationSettings, TrainingSettings


@dataclass(frozen=True)
class PyBulletGeometryValidationConfig:
    training_settings: TrainingSettings
    simulation_settings: SimulationSettings
    seed: int = 500_000
    scenarios: int = 100
    output: Path | None = None
    velocity_aviary_cls: Any | None = None
    drone_model: Any | None = None
    physics: Any | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.scenarios <= 0:
            raise ValueError("scenarios must be positive")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        if not self.training_settings.pybullet_obstacle_randomization.enabled:
            raise ValueError("PyBullet obstacle randomization must be enabled")


def run_pybullet_geometry_validation(config: PyBulletGeometryValidationConfig) -> dict[str, Any]:
    writer = ExperimentArtifactWriter(config.training_settings.artifact)
    stage = config.training_settings.run.stage
    variant = "pybullet_geometry_validation"
    config_hash = _config_hash(config)
    run_id = build_run_id(
        stage=stage,
        variant=variant,
        seed=config.seed,
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(stage, variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json
    env = PyBulletVelocityTrainingEnv(
        simulation_settings=config.simulation_settings,
        settings=config.training_settings.environment,
        enable_pybullet_obstacles=True,
        obstacle_randomization=config.training_settings.pybullet_obstacle_randomization,
        reward_settings=config.training_settings.pybullet_reward,
        curriculum=config.training_settings.pybullet_curriculum,
        velocity_aviary_cls=config.velocity_aviary_cls,
        drone_model=config.drone_model,
        physics=config.physics,
    )
    layouts: list[dict[str, Any]] = []
    expected_phase = (
        config.training_settings.pybullet_curriculum.phases[-1].name
        if config.training_settings.pybullet_curriculum.enabled
        else None
    )
    try:
        for scenario_seed in range(config.seed, config.seed + config.scenarios):
            _, reset_info = env.reset(seed=scenario_seed)
            phase = str(reset_info.get("curriculum_phase", "final"))
            if expected_phase is not None and phase != expected_phase:
                raise RuntimeError("geometry validation must use final curriculum phase")
            initial_clearance = float(reset_info.get("minimum_safety_distance", 0.0))
            if not math.isfinite(initial_clearance):
                raise ValueError("initial minimum safety distance must be finite")
            _, _, _, _, step_info = env.step(
                DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
            )
            layouts.append(
                {
                    "seed": scenario_seed,
                    "curriculum_phase": phase,
                    "obstacle_count": int(reset_info.get("obstacle_count", 0)),
                    "initial_minimum_safety_distance": initial_clearance,
                    "initial_clearance_valid": initial_clearance > env.settings.safety_margin,
                    "first_step_collided": bool(step_info.get("collided", False)),
                }
            )
    finally:
        env.close()

    invalid_initial_clearance_count = sum(not item["initial_clearance_valid"] for item in layouts)
    first_step_collision_count = sum(item["first_step_collided"] for item in layouts)
    geometry_valid = invalid_initial_clearance_count == 0 and first_step_collision_count == 0
    report = {
        "schema_version": 1,
        "record_type": "pybullet_geometry_validation_report",
        "stage": stage,
        "variant": variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lineage": {
            "config_hash": config_hash,
            "source_runner": "swift.experiments.pybullet_geometry_validator.run_pybullet_geometry_validation",
            "validation_backend": "pybullet_final_phase_initial_geometry",
        },
        "runtime": {
            "runtime_contract": "pybullet_velocity_training_compatibility",
            "obstacle_randomization": asdict(config.training_settings.pybullet_obstacle_randomization),
            "curriculum": asdict(config.training_settings.pybullet_curriculum),
        },
        "scenarios": {
            "seed_start": config.seed,
            "seed_end": config.seed + config.scenarios - 1,
            "count": config.scenarios,
        },
        "metrics": {
            "invalid_initial_clearance_count": invalid_initial_clearance_count,
            "first_step_collision_count": first_step_collision_count,
            "minimum_initial_safety_distance": min(
                float(item["initial_minimum_safety_distance"]) for item in layouts
            ),
        },
        "layouts": layouts,
        "readiness": {"geometry_valid": geometry_valid},
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
            outputs=[artifact_reference(output_path, role="geometry_validation_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _config_hash(config: PyBulletGeometryValidationConfig) -> str:
    material = repr(
        (
            config.training_settings,
            config.simulation_settings,
            config.seed,
            config.scenarios,
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
