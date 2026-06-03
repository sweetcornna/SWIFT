import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from swift.config import TrainingRunSettings, TrainingSettings
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig
from swift.experiments.ppo_training_runner import PPOTrainingRunConfig, run_ppo_training_smoke
from swift.rl.ppo import PPOConfig


def test_ppo_training_smoke_returns_json_safe_summary(tmp_path: Path):
    output = tmp_path / "ppo_smoke.json"
    settings = TrainingSettings(
        ppo=PPOConfig(rollout_steps=32, minibatch_size=16, update_epochs=1),
        environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 0.0), max_steps=8),
        run=TrainingRunSettings(stage="stage1", variant="ppo_mlp", seed=3, total_timesteps=128),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
    )

    summary = run_ppo_training_smoke(PPOTrainingRunConfig(settings=settings, output=output))

    assert output.exists()
    raw_summary = output.read_text(encoding="utf-8")
    parsed = json.loads(raw_summary)
    assert parsed == summary
    assert summary["schema_version"] == 1
    assert summary["stage"] == "stage1"
    assert summary["variant"] == "ppo_mlp"
    assert summary["training"]["total_timesteps"] == 128
    assert summary["training"]["updates"] >= 1
    assert summary["artifacts"]["summary_json"] == str(output)
    assert summary["lineage"]["training_backend"] == "torch_ppo_mlp"
    manifest_path = Path(summary["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    history_path = Path(summary["artifacts"]["training_history_jsonl"])
    checkpoint_path = Path(summary["artifacts"]["checkpoint_path"])
    assert history_path.exists()
    assert checkpoint_path.exists()
    assert summary["training"]["history_path"] == str(history_path)
    assert summary["training"]["checkpoint_path"] == str(checkpoint_path)
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_type"] == "experiment_artifact_manifest"
    assert manifest["subject_record_type"] == "ppo_training_smoke"
    assert {reference["role"] for reference in manifest["outputs"]} == {
        "summary_json",
        "training_history_jsonl",
        "checkpoint",
    }
    assert len(history) == summary["training"]["updates"]
    assert {record["record_type"] for record in history} == {"ppo_update"}
    assert "Infinity" not in raw_summary
    assert "NaN" not in raw_summary
    for value in summary["training"].values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_ppo_training_smoke_rejects_non_positive_total_timesteps(tmp_path: Path):
    settings = TrainingSettings(run=TrainingRunSettings(total_timesteps=128))

    with pytest.raises(ValueError, match="total_timesteps"):
        run_ppo_training_smoke(PPOTrainingRunConfig(settings=settings, total_timesteps=0, output=tmp_path / "x.json"))
