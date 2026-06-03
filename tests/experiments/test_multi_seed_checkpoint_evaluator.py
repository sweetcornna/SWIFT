import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from swift.config import TrainingRunSettings, TrainingSettings
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig
from swift.experiments.multi_seed_checkpoint_evaluator import (
    MultiSeedCheckpointEvaluationConfig,
    run_multi_seed_checkpoint_evaluation,
)
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


def _multi_seed_report(tmp_path: Path) -> Path:
    report_path = tmp_path / "multi_seed.json"
    run_multi_seed_training(
        MultiSeedTrainingConfig(
            settings=_settings(tmp_path),
            seeds=(0, 1),
            total_timesteps=32,
            output=report_path,
            evidence_level="long_training_convergence",
        )
    )
    return report_path


def test_multi_seed_checkpoint_evaluation_writes_holdout_report_and_manifest(tmp_path: Path):
    input_report = _multi_seed_report(tmp_path)
    output = tmp_path / "holdout.json"

    report = run_multi_seed_checkpoint_evaluation(
        MultiSeedCheckpointEvaluationConfig(
            input_report=input_report,
            output=output,
            holdout_seeds=(101,),
            episodes=1,
            environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 0.0), max_steps=8),
        )
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["record_type"] == "multi_seed_checkpoint_holdout_report"
    assert report["source_record_type"] == "multi_seed_training_report"
    assert report["training_seeds"] == [0, 1]
    assert report["holdout_seeds"] == [101]
    assert report["seed_count"] == 2
    assert report["holdout_seed_count"] == 1
    assert report["readiness"] == {
        "all_checkpoints_evaluated": True,
        "training_seed_count": 2,
        "holdout_seed_count": 1,
        "convergence_claim": False,
        "evidence_level": "long_training_convergence",
    }
    assert {variant["variant"] for variant in report["variants"]} == {
        "ppo_mlp",
        "ppo_hca",
        "ppo_hca_apf",
    }
    for variant in report["variants"]:
        assert variant["training_seed_count"] == 2
        assert variant["holdout_seed_count"] == 1
        assert variant["evaluation"]["episodes_completed"] == 2
        assert variant["training"]["total_timesteps"] == 32
        assert 0.0 <= variant["metrics"]["success_rate"] <= 1.0
        assert 0.0 <= variant["metrics"]["collision_rate"] <= 1.0
        assert 0.0 <= variant["metrics"]["timeout_rate"] <= 1.0
        assert len(variant["checkpoint_evaluations"]) == 2
        assert all(
            item["policy_family"] in {"ppo_mlp", "ppo_hca", "ppo_hca_apf"}
            for item in variant["checkpoint_evaluations"]
        )

    manifest_path = Path(report["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "multi_seed_checkpoint_holdout_report"
    assert "source_multi_seed_training_report" in {reference["role"] for reference in manifest["inputs"]}
    assert {reference["role"] for reference in manifest["outputs"]} == {"multi_seed_checkpoint_holdout_report"}


def test_multi_seed_checkpoint_evaluation_rejects_holdout_seed_overlap(tmp_path: Path):
    input_report = _multi_seed_report(tmp_path)

    with pytest.raises(ValueError, match="holdout_seeds must be disjoint"):
        run_multi_seed_checkpoint_evaluation(
            MultiSeedCheckpointEvaluationConfig(
                input_report=input_report,
                output=tmp_path / "holdout.json",
                holdout_seeds=(1,),
                episodes=1,
            )
        )
