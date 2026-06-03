import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from swift.config import SimulationSettings, TrainingRunSettings, TrainingSettings
from swift.envs import SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig
from swift.experiments.pybullet_training_runner import (
    PyBulletPPOTrainingRunConfig,
    run_pybullet_ppo_training,
)
from swift.rl.ppo import PPOConfig


def test_pybullet_ppo_training_writes_runtime_contract_summary(tmp_path: Path):
    output = tmp_path / "pybullet_training.json"
    settings = TrainingSettings(
        ppo=PPOConfig(rollout_steps=16, minibatch_size=8, update_epochs=1),
        environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 1.0), max_steps=4, max_speed=1.0),
        run=TrainingRunSettings(stage="stage1", variant="ppo_mlp_pybullet", seed=7, total_timesteps=32),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
    )
    simulation_settings = SimulationSettings(
        pybullet_root=tmp_path / "pybullet",
        pixi_executable=tmp_path / "pybullet" / ".tools" / "pixi" / "pixi.exe",
        required_tasks=("test",),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )

    summary = run_pybullet_ppo_training(
        PyBulletPPOTrainingRunConfig(
            training_settings=settings,
            simulation_settings=simulation_settings,
            output=output,
            enable_pybullet_obstacles=True,
            velocity_aviary_cls=lambda **_: FakeVelocityAviary(),
            drone_model=SimpleNamespace(CF2X="cf2x"),
            physics=SimpleNamespace(PYB="pyb"),
        )
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == summary
    assert summary["record_type"] == "pybullet_ppo_training_report"
    assert summary["variant"] == "ppo_mlp_pybullet"
    assert summary["lineage"]["training_backend"] == "torch_ppo_mlp_pybullet_velocity"
    assert summary["lineage"]["environment_adapter"] == "swift.envs.PyBulletVelocityTrainingEnv"
    assert summary["runtime"]["runtime_contract"] == "pybullet_velocity_training_compatibility"
    assert summary["runtime"]["pybullet_root"] == str(simulation_settings.pybullet_root)
    assert summary["runtime"]["enable_pybullet_obstacles"] is True
    assert summary["training"]["total_timesteps"] == 32
    assert summary["training"]["updates"] >= 1
    assert summary["artifacts"]["summary_json"] == str(output)
    assert summary["training"]["history_path"] == summary["artifacts"]["training_history_jsonl"]
    assert summary["training"]["checkpoint_path"] == summary["artifacts"]["checkpoint_path"]
    assert Path(summary["artifacts"]["training_history_jsonl"]).exists()
    assert Path(summary["artifacts"]["checkpoint_path"]).exists()
    manifest = json.loads(Path(summary["artifacts"]["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "pybullet_ppo_training_report"
    assert {reference["role"] for reference in manifest["outputs"]} == {
        "summary_json",
        "training_history_jsonl",
        "checkpoint",
    }
    for reference in manifest["outputs"]:
        assert len(reference["sha256"]) == 64
        assert reference["bytes"] > 0
    assert "Infinity" not in output.read_text(encoding="utf-8")
    assert "NaN" not in output.read_text(encoding="utf-8")
    for value in summary["training"].values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_pybullet_ppo_training_rejects_non_positive_total_timesteps(tmp_path: Path):
    settings = TrainingSettings(run=TrainingRunSettings(total_timesteps=128))
    simulation_settings = SimulationSettings(
        pybullet_root=tmp_path,
        pixi_executable=tmp_path / "pixi.exe",
        required_tasks=("test",),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )

    with pytest.raises(ValueError, match="total_timesteps"):
        run_pybullet_ppo_training(
            PyBulletPPOTrainingRunConfig(
                training_settings=settings,
                simulation_settings=simulation_settings,
                total_timesteps=0,
                output=tmp_path / "x.json",
            )
        )


class FakeVelocityAviary:
    def reset(self, seed=None, options=None):
        return [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]], {
            "seed": seed
        }

    def step(self, action):
        return (
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0]],
            -1.0,
            False,
            False,
            {"source": "fake"},
        )

    def close(self):
        pass
