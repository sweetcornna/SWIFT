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


def _settings(
    tmp_path: Path,
    *,
    goal: tuple[float, float, float] = (0.5, 0.0, 0.1125),
) -> TrainingSettings:
    return TrainingSettings(
        environment=SimpleAvoidanceSettings(
            start=(0.0, 0.0, 0.1125),
            goal=goal,
            safety_margin=0.1,
        ),
        run=TrainingRunSettings(
            stage="stage1",
            variant="ppo_mlp_pybullet_randomized",
            seed=0,
            total_timesteps=100000,
        ),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
        pybullet_obstacle_randomization=PyBulletObstacleRandomizationSettings(enabled=True),
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


def _write_training_report(tmp_path: Path, *, seeds: tuple[int, ...] = (8, 9, 10)) -> Path:
    from swift.experiments.pybullet_task_contract import build_pybullet_task_contract

    task_contract = build_pybullet_task_contract(_settings(tmp_path))
    seed_runs = []
    for seed in seeds:
        checkpoint = tmp_path / f"seed_{seed}.ckpt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        seed_runs.append(
            {
                "seed": seed,
                "completed": True,
                "training": {"total_timesteps": 100000, "episodes_completed": 100},
                "metrics": {
                    "success_rate": 1.0,
                    "collision_rate": 0.0,
                    "timeout_rate": 0.0,
                    "average_episode_return": 10.0,
                },
                "artifacts": {"checkpoint_path": str(checkpoint)},
            }
        )
    path = tmp_path / "training.json"
    path.write_text(
        json.dumps(
            {
                "record_type": "pybullet_multi_seed_training_report",
                "run_id": "training-run",
                "lineage": {"task_contract_hash": task_contract["hash"]},
                "task_contract": task_contract,
                "seeds": list(seeds),
                "seed_count": len(seeds),
                "total_timesteps": 100000,
                "seed_runs": seed_runs,
                "readiness": {"all_seeds_completed": True},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_multi_seed_holdout_evaluates_same_scenarios_and_aggregates_worst_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swift.experiments.pybullet_multi_seed_checkpoint_evaluator as evaluator
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
        run_pybullet_multi_seed_checkpoint_evaluation,
    )

    training_report = _write_training_report(tmp_path)
    calls = []
    metrics = {
        8: (0.98, 0.0, 0.02, 0.14),
        9: (0.95, 0.0, 0.05, 0.11),
        10: (0.97, 0.0, 0.03, 0.13),
    }

    def fake_checkpoint_eval(config):
        training_seed = int(config.checkpoint_path.stem.removeprefix("seed_"))
        calls.append((training_seed, config))
        success, collision, timeout, safety = metrics[training_seed]
        assert config.output is not None
        config.output.parent.mkdir(parents=True, exist_ok=True)
        manifest = config.output.with_suffix(".manifest.json")
        manifest.write_text("{}\n", encoding="utf-8")
        report = {
            "record_type": "pybullet_ppo_checkpoint_evaluation",
            "run_id": f"eval-{training_seed}",
            "checkpoint_path": str(config.checkpoint_path),
            "episodes_requested": config.episodes,
            "seed": config.seed,
            "metrics": {
                "success_rate": success,
                "collision_rate": collision,
                "timeout_rate": timeout,
                "average_minimum_safety_distance": safety,
                "average_episode_return": 20.0,
            },
            "artifacts": {
                "summary_json": str(config.output),
                "manifest_json": str(manifest),
                "checkpoint_path": str(config.checkpoint_path),
            },
        }
        config.output.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(evaluator, "run_pybullet_checkpoint_evaluation", fake_checkpoint_eval)
    output = tmp_path / "holdout.json"

    report = run_pybullet_multi_seed_checkpoint_evaluation(
        PyBulletMultiSeedCheckpointEvaluationConfig(
            training_report_path=training_report,
            training_settings=_settings(tmp_path),
            simulation_settings=_simulation_settings(tmp_path),
            episodes_per_checkpoint=100,
            holdout_seed=1000000,
            output=output,
        )
    )

    assert [seed for seed, _ in calls] == [8, 9, 10]
    assert all(config.seed == 1000000 for _, config in calls)
    assert all(config.episodes == 100 for _, config in calls)
    assert all(config.enable_pybullet_obstacles for _, config in calls)
    assert report["record_type"] == "pybullet_multi_seed_checkpoint_holdout_report"
    assert report["holdout"]["seed_start"] == 1000000
    assert report["holdout"]["seed_end"] == 1000099
    assert report["holdout"]["kind"] == "consumed"
    assert report["holdout"]["episodes_per_checkpoint"] == 100
    assert report["lineage"]["task_contract_hash"] == report["task_contract"]["hash"]
    assert report["metrics"]["worst_success_rate"] == pytest.approx(0.95)
    assert report["metrics"]["max_collision_rate"] == pytest.approx(0.0)
    assert report["metrics"]["max_timeout_rate"] == pytest.approx(0.05)
    assert report["metrics"]["worst_average_minimum_safety_distance"] == pytest.approx(0.11)
    assert report["readiness"]["all_checkpoints_evaluated"] is True
    assert len(report["checkpoint_evaluations"]) == 3
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert {item["role"] for item in manifest["inputs"]} == {
        "multi_seed_training_report",
        "train_seed_8_checkpoint_evaluation",
        "train_seed_9_checkpoint_evaluation",
        "train_seed_10_checkpoint_evaluation",
    }


def test_multi_seed_holdout_rejects_training_seed_overlap(tmp_path: Path) -> None:
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
        run_pybullet_multi_seed_checkpoint_evaluation,
    )

    config = PyBulletMultiSeedCheckpointEvaluationConfig(
        training_report_path=_write_training_report(tmp_path, seeds=(8, 1000000)),
        training_settings=_settings(tmp_path),
        simulation_settings=_simulation_settings(tmp_path),
        episodes_per_checkpoint=100,
        holdout_seed=1000000,
        output=tmp_path / "holdout.json",
    )

    with pytest.raises(ValueError, match="holdout seeds must not overlap training seeds"):
        run_pybullet_multi_seed_checkpoint_evaluation(config)


def test_multi_seed_holdout_defaults_to_development_seed_range(tmp_path: Path) -> None:
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
    )

    config = PyBulletMultiSeedCheckpointEvaluationConfig(
        training_report_path=_write_training_report(tmp_path),
        training_settings=_settings(tmp_path),
        simulation_settings=_simulation_settings(tmp_path),
    )

    assert config.holdout_seed == 500000


def test_multi_seed_holdout_rejects_task_contract_drift_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swift.experiments.pybullet_multi_seed_checkpoint_evaluator as evaluator
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
        run_pybullet_multi_seed_checkpoint_evaluation,
    )

    monkeypatch.setattr(
        evaluator,
        "run_pybullet_checkpoint_evaluation",
        lambda config: pytest.fail("checkpoint evaluation must not run after task-contract drift"),
    )

    with pytest.raises(ValueError, match="task contract does not match"):
        run_pybullet_multi_seed_checkpoint_evaluation(
            PyBulletMultiSeedCheckpointEvaluationConfig(
                training_report_path=_write_training_report(tmp_path),
                training_settings=_settings(tmp_path, goal=(0.6, 0.0, 0.1125)),
                simulation_settings=_simulation_settings(tmp_path),
                output=tmp_path / "holdout.json",
            )
        )


def test_multi_seed_holdout_rejects_training_report_without_task_contract(tmp_path: Path) -> None:
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
        run_pybullet_multi_seed_checkpoint_evaluation,
    )

    training_report = _write_training_report(tmp_path)
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    payload.pop("task_contract")
    payload["lineage"].pop("task_contract_hash")
    training_report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training report must contain task contract"):
        run_pybullet_multi_seed_checkpoint_evaluation(
            PyBulletMultiSeedCheckpointEvaluationConfig(
                training_report_path=training_report,
                training_settings=_settings(tmp_path),
                simulation_settings=_simulation_settings(tmp_path),
            )
        )


def test_multi_seed_holdout_rejects_tampered_task_contract_settings(tmp_path: Path) -> None:
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
        run_pybullet_multi_seed_checkpoint_evaluation,
    )

    training_report = _write_training_report(tmp_path)
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    payload["task_contract"]["settings"]["environment"]["goal"] = [0.6, 0.0, 0.1125]
    training_report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="task contract settings do not match hash"):
        run_pybullet_multi_seed_checkpoint_evaluation(
            PyBulletMultiSeedCheckpointEvaluationConfig(
                training_report_path=training_report,
                training_settings=_settings(tmp_path),
                simulation_settings=_simulation_settings(tmp_path),
            )
        )


def test_multi_seed_holdout_rejects_wrong_training_record_type(tmp_path: Path) -> None:
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
        run_pybullet_multi_seed_checkpoint_evaluation,
    )

    training_report = _write_training_report(tmp_path)
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    payload["record_type"] = "pybullet_ppo_training_report"
    training_report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pybullet_multi_seed_training_report"):
        run_pybullet_multi_seed_checkpoint_evaluation(
            PyBulletMultiSeedCheckpointEvaluationConfig(
                training_report_path=training_report,
                training_settings=_settings(tmp_path),
                simulation_settings=_simulation_settings(tmp_path),
            )
        )
