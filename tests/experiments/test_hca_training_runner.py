import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from swift.config import TrainingRunSettings, TrainingSettings
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig
from swift.experiments.hca_training_runner import HCATrainingRunConfig, run_hca_training_smoke
from swift.rl.ppo import PPOConfig


def test_hca_training_smoke_returns_json_safe_summary(tmp_path: Path):
    output = tmp_path / "hca_smoke.json"
    settings = TrainingSettings(
        ppo=PPOConfig(rollout_steps=32, minibatch_size=16, update_epochs=1),
        environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 0.0), max_steps=8),
        run=TrainingRunSettings(stage="stage2", variant="ppo_hca", seed=3, total_timesteps=64),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
    )

    summary = run_hca_training_smoke(HCATrainingRunConfig(settings=settings, output=output))

    raw_summary = output.read_text(encoding="utf-8")
    assert json.loads(raw_summary) == summary
    assert summary["schema_version"] == 1
    assert summary["record_type"] == "ppo_hca_training_smoke"
    assert summary["stage"] == "stage2"
    assert summary["variant"] == "ppo_hca"
    assert summary["training"]["total_timesteps"] == 64
    assert summary["training"]["updates"] >= 1
    history_path = Path(summary["artifacts"]["training_history_jsonl"])
    checkpoint_path = Path(summary["artifacts"]["checkpoint_path"])
    assert history_path.exists()
    assert checkpoint_path.exists()
    assert "NaN" not in raw_summary
    assert "Infinity" not in raw_summary
    for value in summary["training"].values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_hca_training_smoke_rejects_non_positive_total_timesteps(tmp_path: Path):
    settings = TrainingSettings(run=TrainingRunSettings(total_timesteps=128))

    with pytest.raises(ValueError, match="total_timesteps"):
        run_hca_training_smoke(HCATrainingRunConfig(settings=settings, total_timesteps=0, output=tmp_path / "x.json"))
