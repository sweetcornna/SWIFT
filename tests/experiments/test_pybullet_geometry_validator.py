from __future__ import annotations

from pathlib import Path

import pytest

from swift.config import (
    PyBulletCurriculumPhaseSettings,
    PyBulletCurriculumSettings,
    PyBulletObstacleRandomizationSettings,
    SimulationSettings,
    TrainingSettings,
)
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig


def test_pybullet_geometry_validation_reports_valid_final_phase_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import swift.experiments.pybullet_geometry_validator as validator
    except ModuleNotFoundError:
        pytest.fail("pybullet geometry validator is not implemented")

    training_settings = _training_settings(tmp_path)
    simulation_settings = _simulation_settings(tmp_path)

    class FakeGeometryEnv:
        settings = training_settings.environment

        def __init__(self, **kwargs):
            del kwargs

        def reset(self, seed=None):
            return (0.0,) * 15, {
                "scenario_seed": seed,
                "curriculum_phase": "randomized_final",
                "obstacle_count": 1,
                "minimum_safety_distance": 0.12,
            }

        def step(self, action):
            del action
            return (0.0,) * 15, 0.0, False, False, {"collided": False}

        def close(self):
            pass

    monkeypatch.setattr(validator, "PyBulletVelocityTrainingEnv", FakeGeometryEnv)
    output_path = tmp_path / "geometry.json"
    report = validator.run_pybullet_geometry_validation(
        validator.PyBulletGeometryValidationConfig(
            training_settings=training_settings,
            simulation_settings=simulation_settings,
            seed=500000,
            scenarios=3,
            output=output_path,
        )
    )

    assert report["record_type"] == "pybullet_geometry_validation_report"
    assert report["scenarios"] == {"seed_start": 500000, "seed_end": 500002, "count": 3}
    assert report["metrics"]["invalid_initial_clearance_count"] == 0
    assert report["metrics"]["first_step_collision_count"] == 0
    assert report["readiness"]["geometry_valid"] is True
    assert all(item["curriculum_phase"] == "randomized_final" for item in report["layouts"])
    assert output_path.is_file()
    assert Path(report["artifacts"]["manifest_json"]).is_file()


def test_pybullet_geometry_validation_rejects_clearance_at_safety_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import swift.experiments.pybullet_geometry_validator as validator
    except ModuleNotFoundError:
        pytest.fail("pybullet geometry validator is not implemented")

    training_settings = _training_settings(tmp_path)

    class BoundaryGeometryEnv:
        settings = training_settings.environment

        def __init__(self, **kwargs):
            del kwargs

        def reset(self, seed=None):
            return (0.0,) * 15, {
                "scenario_seed": seed,
                "curriculum_phase": "randomized_final",
                "obstacle_count": 1,
                "minimum_safety_distance": 0.1,
            }

        def step(self, action):
            del action
            return (0.0,) * 15, 0.0, False, False, {"collided": False}

        def close(self):
            pass

    monkeypatch.setattr(validator, "PyBulletVelocityTrainingEnv", BoundaryGeometryEnv)
    report = validator.run_pybullet_geometry_validation(
        validator.PyBulletGeometryValidationConfig(
            training_settings=training_settings,
            simulation_settings=_simulation_settings(tmp_path),
            seed=500100,
            scenarios=1,
            output=tmp_path / "boundary.json",
        )
    )

    assert report["metrics"]["invalid_initial_clearance_count"] == 1
    assert report["readiness"]["geometry_valid"] is False


def _training_settings(tmp_path: Path) -> TrainingSettings:
    return TrainingSettings(
        environment=SimpleAvoidanceSettings(safety_margin=0.1),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
        pybullet_obstacle_randomization=PyBulletObstacleRandomizationSettings(enabled=True),
        pybullet_curriculum=PyBulletCurriculumSettings(
            enabled=True,
            phases=(
                PyBulletCurriculumPhaseSettings("randomized_final", 1.0, 1, 3, True),
            ),
        ),
    )


def _simulation_settings(tmp_path: Path) -> SimulationSettings:
    return SimulationSettings(
        pybullet_root=tmp_path,
        pixi_executable=tmp_path / "pixi.exe",
        required_tasks=(),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )
