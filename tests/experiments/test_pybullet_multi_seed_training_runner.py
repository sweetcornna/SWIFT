from __future__ import annotations

import json
from pathlib import Path

import pytest

from swift.config import (
    PyBulletObstacleRandomizationSettings,
    SimulationSettings,
    TrainingRunSettings,
    TrainingSettings,
)
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig


def _settings(tmp_path: Path, *, randomization_enabled: bool = True) -> TrainingSettings:
    return TrainingSettings(
        environment=SimpleAvoidanceSettings(
            start=(0.0, 0.0, 0.1125),
            goal=(0.5, 0.0, 0.1125),
            safety_margin=0.1,
        ),
        run=TrainingRunSettings(
            stage="stage1",
            variant="ppo_mlp_pybullet_randomized",
            seed=0,
            total_timesteps=32,
        ),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
        pybullet_obstacle_randomization=PyBulletObstacleRandomizationSettings(
            enabled=randomization_enabled
        ),
    )


def _simulation_settings(tmp_path: Path) -> SimulationSettings:
    return SimulationSettings(
        pybullet_root=tmp_path / "pybullet",
        pixi_executable=tmp_path / "pybullet" / "pixi.exe",
        required_tasks=("test",),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )


def test_pybullet_multi_seed_training_writes_child_runs_and_worst_case_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swift.experiments.pybullet_multi_seed_training_runner as runner
    from swift.experiments.pybullet_multi_seed_training_runner import (
        PyBulletMultiSeedTrainingConfig,
        run_pybullet_multi_seed_training,
    )

    calls = []
    per_seed = {
        8: {"success_rate": 0.98, "collision_rate": 0.0, "timeout_rate": 0.02},
        9: {"success_rate": 0.96, "collision_rate": 0.0, "timeout_rate": 0.04},
        10: {"success_rate": 0.97, "collision_rate": 0.0, "timeout_rate": 0.03},
    }

    def fake_single_seed(config):
        calls.append(config)
        metrics = per_seed[config.seed]
        assert config.output is not None
        checkpoint = config.output.with_suffix(".ckpt")
        history = config.output.with_suffix(".jsonl")
        manifest = config.output.with_suffix(".manifest.json")
        config.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{config.seed}".encode())
        history.write_text("{}\n", encoding="utf-8")
        manifest.write_text("{}\n", encoding="utf-8")
        report = {
            "record_type": "pybullet_ppo_training_report",
            "run_id": f"seed-{config.seed}",
            "lineage": {"config_hash": f"config-{config.seed}"},
            "runtime": {"runtime_contract": "pybullet_velocity_training_compatibility"},
            "metrics": {**metrics, "average_episode_return": 10.0 + config.seed},
            "training": {
                **metrics,
                "average_episode_return": 10.0 + config.seed,
                "total_timesteps": config.total_timesteps,
                "updates": 2,
                "episodes_completed": 4,
                "checkpoint_path": str(checkpoint),
                "history_path": str(history),
            },
            "artifacts": {
                "summary_json": str(config.output),
                "manifest_json": str(manifest),
                "checkpoint_path": str(checkpoint),
                "training_history_jsonl": str(history),
            },
        }
        config.output.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(runner, "run_pybullet_ppo_training", fake_single_seed)
    output = tmp_path / "outputs" / "training" / "pybullet_randomized_3x100k.json"

    report = run_pybullet_multi_seed_training(
        PyBulletMultiSeedTrainingConfig(
            training_settings=_settings(tmp_path),
            simulation_settings=_simulation_settings(tmp_path),
            seeds=(8, 9, 10),
            total_timesteps=100000,
            output=output,
        )
    )

    assert [call.seed for call in calls] == [8, 9, 10]
    assert all(call.total_timesteps == 100000 for call in calls)
    assert all(call.enable_pybullet_obstacles for call in calls)
    assert [call.output.name for call in calls] == ["seed_8.json", "seed_9.json", "seed_10.json"]
    assert report["record_type"] == "pybullet_multi_seed_training_report"
    assert report["seeds"] == [8, 9, 10]
    assert report["seed_count"] == 3
    assert report["metrics"]["worst_success_rate"] == pytest.approx(0.96)
    assert report["metrics"]["max_collision_rate"] == pytest.approx(0.0)
    assert report["metrics"]["max_timeout_rate"] == pytest.approx(0.04)
    assert report["readiness"]["all_seeds_completed"] is True
    assert len(report["seed_runs"]) == 3
    assert report["seed_runs"][0]["lineage"]["config_hash"] == "config-8"
    assert report["seed_runs"][0]["runtime"]["runtime_contract"] == (
        "pybullet_velocity_training_compatibility"
    )
    assert report["lineage"]["task_contract_hash"] == report["task_contract"]["hash"]
    assert report["task_contract"]["settings"]["environment"]["goal"] == [0.5, 0.0, 0.1125]
    assert json.loads(output.read_text(encoding="utf-8")) == report
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert {item["role"] for item in manifest["inputs"]} == {
        "seed_8_training_report",
        "seed_9_training_report",
        "seed_10_training_report",
    }


def test_pybullet_multi_seed_training_rejects_duplicate_seeds(tmp_path: Path) -> None:
    from swift.experiments.pybullet_multi_seed_training_runner import PyBulletMultiSeedTrainingConfig

    with pytest.raises(ValueError, match="seeds must be unique"):
        PyBulletMultiSeedTrainingConfig(
            training_settings=_settings(tmp_path),
            simulation_settings=_simulation_settings(tmp_path),
            seeds=(8, 8),
        )


def test_pybullet_multi_seed_training_requires_randomization(tmp_path: Path) -> None:
    from swift.experiments.pybullet_multi_seed_training_runner import PyBulletMultiSeedTrainingConfig

    with pytest.raises(ValueError, match="randomization must be enabled"):
        PyBulletMultiSeedTrainingConfig(
            training_settings=_settings(tmp_path, randomization_enabled=False),
            simulation_settings=_simulation_settings(tmp_path),
        )
