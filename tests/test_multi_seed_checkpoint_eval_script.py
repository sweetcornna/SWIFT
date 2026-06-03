import json
import subprocess
import sys
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


def test_multi_seed_checkpoint_eval_script_writes_holdout_report(tmp_path: Path):
    input_report = _multi_seed_report(tmp_path)
    output = tmp_path / "holdout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multi_seed_checkpoint_eval.py",
            "--input",
            str(input_report),
            "--episodes",
            "1",
            "--holdout-seeds",
            "101",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["record_type"] == "multi_seed_checkpoint_holdout_report"
    assert report["holdout_seeds"] == [101]
    assert "SWIFT multi-seed checkpoint holdout written" in result.stdout
    assert "variants=3 training_seeds=2 checkpoints=6" in result.stdout


def test_multi_seed_checkpoint_eval_script_dry_run_validates_input(tmp_path: Path):
    input_report = _multi_seed_report(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multi_seed_checkpoint_eval.py",
            "--input",
            str(input_report),
            "--holdout-seeds",
            "101",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT multi-seed checkpoint holdout config OK" in result.stdout
