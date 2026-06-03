from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from swift.config import TrainingRunSettings, TrainingSettings
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig
from swift.experiments.multi_seed_training_runner import (
    MultiSeedTrainingConfig,
    run_multi_seed_training,
)
from swift.rl.ppo import PPOConfig


def _settings(tmp_path: Path) -> TrainingSettings:
    return TrainingSettings(
        ppo=PPOConfig(rollout_steps=16, minibatch_size=8, update_epochs=1),
        environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 0.0), max_steps=8),
        run=TrainingRunSettings(stage="stage4", variant="multi_seed", seed=0, total_timesteps=32),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
    )


def test_multi_seed_training_runs_seed_ablation_reports_and_aggregates_worst_case_metrics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "multi_seed.json"

    report = run_multi_seed_training(
        MultiSeedTrainingConfig(
            settings=_settings(tmp_path),
            seeds=(0, 1),
            total_timesteps=32,
            output=output,
            evidence_level="long_training_convergence",
        )
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["record_type"] == "multi_seed_training_report"
    assert report["seed_count"] == 2
    assert report["seeds"] == [0, 1]
    assert report["readiness"] == {
        "all_seeds_completed": True,
        "seed_count": 2,
        "required_variants": 3,
        "convergence_claim": False,
        "evidence_level": "long_training_convergence",
    }
    assert len(report["seed_reports"]) == 2
    assert {seed_report["seed"] for seed_report in report["seed_reports"]} == {0, 1}
    assert {variant["variant"] for variant in report["variants"]} == {
        "ppo_mlp",
        "ppo_hca",
        "ppo_hca_apf",
    }
    for variant in report["variants"]:
        assert variant["completed_seeds"] == 2
        assert variant["seed_count"] == 2
        assert 0.0 <= variant["worst_success_rate"] <= 1.0
        assert 0.0 <= variant["max_collision_rate"] <= 1.0
        assert 0.0 <= variant["max_timeout_rate"] <= 1.0
        assert variant["min_seed_total_timesteps"] == 32
        assert variant["aggregate_total_timesteps"] == 64
        assert variant["metrics"]["success_rate"] == variant["worst_success_rate"]
        assert variant["metrics"]["collision_rate"] == variant["max_collision_rate"]
        assert variant["metrics"]["timeout_rate"] == variant["max_timeout_rate"]
        assert variant["training"]["total_timesteps"] == variant["min_seed_total_timesteps"]
        assert variant["training"]["episodes_completed"] == variant["episodes_completed"]
        assert len(variant["seed_metrics"]) == 2

    manifest_path = Path(report["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "multi_seed_training_report"
    assert {reference["role"] for reference in manifest["outputs"]} == {"multi_seed_training_report"}
    assert {reference["role"] for reference in manifest["inputs"]} == {
        "seed_0_ablation_report",
        "seed_1_ablation_report",
    }


def test_multi_seed_training_rejects_empty_or_duplicate_seed_sets(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="seeds must contain at least one seed"):
        MultiSeedTrainingConfig(settings=settings, seeds=())

    with pytest.raises(ValueError, match="seeds must be unique"):
        MultiSeedTrainingConfig(settings=settings, seeds=(0, 0))
