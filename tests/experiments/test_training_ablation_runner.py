import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from swift.config import TrainingRunSettings, TrainingSettings
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig
from swift.experiments.training_ablation_runner import (
    TrainingAblationConfig,
    run_training_ablation,
    score_ablation_metrics,
)
from swift.rl.ppo import PPOConfig


def _settings(tmp_path: Path) -> TrainingSettings:
    return TrainingSettings(
        ppo=PPOConfig(rollout_steps=16, minibatch_size=8, update_epochs=1),
        environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 0.0), max_steps=8),
        run=TrainingRunSettings(stage="stage4", variant="ablation", seed=5, total_timesteps=32),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
    )


def test_training_ablation_runs_three_variants_and_writes_ranked_manifest(tmp_path: Path):
    output = tmp_path / "ablation.json"

    report = run_training_ablation(
        TrainingAblationConfig(settings=_settings(tmp_path), total_timesteps=32, output=output)
    )

    raw_report = output.read_text(encoding="utf-8")
    assert json.loads(raw_report) == report
    assert report["schema_version"] == 1
    assert report["record_type"] == "training_ablation_report"
    assert report["stage"] == "stage4"
    assert report["variant"] == "ablation"
    assert report["readiness"] == {
        "all_variants_completed": True,
        "completed_variants": 3,
        "required_variants": 3,
        "convergence_claim": False,
        "evidence_level": "cpu_smoke_ablation",
    }
    assert report["ranking_formula"] == {
        "score": "100*success_rate - 100*collision_rate - 10*timeout_rate + 0.01*average_episode_return",
        "sort": "score_desc_variant_asc",
        "convergence_claim": False,
    }
    assert [variant["variant"] for variant in report["variants"]] == [
        "ppo_mlp",
        "ppo_hca",
        "ppo_hca_apf",
    ]
    assert [entry["rank"] for entry in report["ranking"]] == [1, 2, 3]
    assert [entry["score"] for entry in report["ranking"]] == sorted(
        [entry["score"] for entry in report["ranking"]],
        reverse=True,
    )
    assert report["best_variant"] == report["ranking"][0]["variant"]
    assert "NaN" not in raw_report
    assert "Infinity" not in raw_report

    manifest_path = Path(report["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "training_ablation_report"
    assert {reference["role"] for reference in manifest["inputs"]} == {
        "ppo_mlp_summary",
        "ppo_hca_summary",
        "ppo_hca_apf_summary",
    }
    assert {reference["role"] for reference in manifest["outputs"]} == {"ablation_report"}

    for variant in report["variants"]:
        assert variant["completed"] is True
        assert variant["training"]["total_timesteps"] == 32
        assert Path(variant["artifacts"]["summary_json"]).exists()
        assert Path(variant["artifacts"]["manifest_json"]).exists()
        assert math.isfinite(variant["score"])
        summary = json.loads(Path(variant["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
        assert summary["stage"] == "stage4"
        assert summary["variant"] == variant["variant"]


def test_score_ablation_metrics_rewards_success_and_penalizes_collisions():
    safe_success = {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "average_episode_return": 5.0,
    }
    collision = {
        "success_rate": 0.0,
        "collision_rate": 1.0,
        "timeout_rate": 0.0,
        "average_episode_return": 5.0,
    }
    timeout = {
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "timeout_rate": 1.0,
        "average_episode_return": 5.0,
    }

    assert score_ablation_metrics(safe_success) > score_ablation_metrics(timeout)
    assert score_ablation_metrics(timeout) > score_ablation_metrics(collision)


def test_training_ablation_rejects_non_positive_total_timesteps(tmp_path: Path):
    with pytest.raises(ValueError, match="total_timesteps"):
        TrainingAblationConfig(settings=_settings(tmp_path), total_timesteps=0)


def test_training_ablation_can_mark_long_training_evidence_level(tmp_path: Path):
    output = tmp_path / "long_ablation.json"

    report = run_training_ablation(
        TrainingAblationConfig(
            settings=_settings(tmp_path),
            total_timesteps=32,
            output=output,
            evidence_level="long_training_convergence",
        )
    )

    assert report["lineage"]["evidence_level"] == "long_training_convergence"
    assert report["readiness"]["evidence_level"] == "long_training_convergence"
    assert report["readiness"]["convergence_claim"] is False
