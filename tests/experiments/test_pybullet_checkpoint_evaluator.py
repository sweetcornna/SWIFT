import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from swift.config import SimulationSettings, TrainingRunSettings, TrainingSettings
from swift.envs import PyBulletVelocityTrainingEnv, SimpleAvoidanceSettings
from swift.experiments import ExperimentArtifactConfig
from swift.experiments.pybullet_checkpoint_evaluator import (
    PyBulletCheckpointEvaluationConfig,
    run_pybullet_checkpoint_evaluation,
)
from swift.rl.ppo import MLPActorCriticConfig, PPOConfig, PPOTrainingConfig
from swift.rl.hca import HCAActorCriticConfig
from swift.rl.ppo import HCAPPOTrainingConfig
from swift.rl.torch_hca_ppo import train_ppo_hca
from swift.rl.torch_ppo import train_ppo_mlp


def test_pybullet_checkpoint_evaluation_writes_summary_and_manifest(tmp_path: Path):
    checkpoint_path = tmp_path / "ppo.ckpt"
    output_path = tmp_path / "pybullet_checkpoint_eval.json"
    training_config_path = tmp_path / "training.yaml"
    simulation_config_path = tmp_path / "simulation.yaml"
    training_config_path.write_text("run:\n  variant: ppo_mlp_pybullet\n", encoding="utf-8")
    simulation_config_path.write_text("pybullet_root: pybullet\n", encoding="utf-8")
    training_settings = TrainingSettings(
        ppo=PPOConfig(rollout_steps=16, minibatch_size=8, update_epochs=1),
        environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 1.0), max_steps=4, max_speed=1.0),
        run=TrainingRunSettings(stage="stage1", variant="ppo_mlp_pybullet", seed=13, total_timesteps=32),
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

    train_ppo_mlp(
        lambda: PyBulletVelocityTrainingEnv(
            simulation_settings=simulation_settings,
            settings=training_settings.environment,
            velocity_aviary_cls=lambda **_: FakeVelocityAviary(),
            drone_model=SimpleNamespace(CF2X="cf2x"),
            physics=SimpleNamespace(PYB="pyb"),
        ),
        PPOTrainingConfig(
            total_timesteps=32,
            rollout_steps=16,
            minibatch_size=8,
            update_epochs=1,
            torch_num_threads=1,
            seed=13,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
            checkpoint_path=checkpoint_path,
        ),
    )

    summary = run_pybullet_checkpoint_evaluation(
        PyBulletCheckpointEvaluationConfig(
            checkpoint_path=checkpoint_path,
            training_settings=training_settings,
            simulation_settings=simulation_settings,
            training_config_path=training_config_path,
            simulation_config_path=simulation_config_path,
            output=output_path,
            episodes=2,
            seed=101,
            enable_pybullet_obstacles=True,
            velocity_aviary_cls=lambda **_: FakeVelocityAviary(),
            drone_model=SimpleNamespace(CF2X="cf2x"),
            physics=SimpleNamespace(PYB="pyb"),
        )
    )

    raw_summary = output_path.read_text(encoding="utf-8")
    assert json.loads(raw_summary) == summary
    assert summary["record_type"] == "pybullet_ppo_checkpoint_evaluation"
    assert summary["stage"] == "stage1"
    assert summary["variant"] == "ppo_mlp_pybullet_pybullet_eval"
    assert summary["run_id"].startswith("stage1_ppo_mlp_pybullet_pybullet_eval_seed-101_cfg-")
    assert summary["checkpoint_path"] == str(checkpoint_path)
    assert summary["policy_family"] == "ppo_mlp"
    assert summary["episodes_requested"] == 2
    assert len(summary["episodes"]) == 2
    assert summary["runtime"]["runtime_contract"] == "pybullet_velocity_training_compatibility"
    assert summary["runtime"]["enable_pybullet_obstacles"] is True
    assert summary["lineage"]["environment_adapter"] == "swift.envs.PyBulletVelocityTrainingEnv"
    assert summary["artifacts"]["summary_json"] == str(output_path)
    assert summary["artifacts"]["training_config_yaml"] == str(training_config_path)
    assert summary["artifacts"]["simulation_config_yaml"] == str(simulation_config_path)
    manifest_path = Path(summary["artifacts"]["manifest_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "pybullet_ppo_checkpoint_evaluation"
    assert manifest["run_id"] == summary["run_id"]
    assert manifest["stage"] == summary["stage"]
    assert manifest["variant"] == summary["variant"]
    assert {reference["role"] for reference in manifest["inputs"]} == {
        "checkpoint",
        "training_config_yaml",
        "simulation_config_yaml",
    }
    assert {reference["role"] for reference in manifest["outputs"]} == {"summary_json"}
    for reference in [*manifest["inputs"], *manifest["outputs"]]:
        assert len(reference["sha256"]) == 64
        assert reference["bytes"] > 0
    for value in summary["metrics"].values():
        if isinstance(value, float):
            assert math.isfinite(value)
    assert "NaN" not in raw_summary
    assert "Infinity" not in raw_summary


def test_pybullet_checkpoint_evaluation_rejects_non_positive_episode_count(tmp_path: Path):
    with pytest.raises(ValueError, match="episodes"):
        PyBulletCheckpointEvaluationConfig(
            checkpoint_path=tmp_path / "missing.ckpt",
            training_settings=TrainingSettings(),
            simulation_settings=SimulationSettings(
                pybullet_root=tmp_path,
                pixi_executable=tmp_path / "pixi.exe",
                required_tasks=("test",),
                check_task="test",
                smoke_task="drone-demo",
                command_timeout_seconds=30,
            ),
            episodes=0,
        )


def test_pybullet_checkpoint_evaluation_rejects_hca_checkpoint_until_pybullet_hca_is_tested(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "hca.ckpt"
    training_settings = TrainingSettings(
        environment=SimpleAvoidanceSettings(goal=(4.0, 0.0, 1.0), max_steps=4, max_speed=1.0),
    )
    simulation_settings = SimulationSettings(
        pybullet_root=tmp_path / "pybullet",
        pixi_executable=tmp_path / "pybullet" / ".tools" / "pixi" / "pixi.exe",
        required_tasks=("test",),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )

    train_ppo_hca(
        lambda: PyBulletVelocityTrainingEnv(
            simulation_settings=simulation_settings,
            settings=training_settings.environment,
            velocity_aviary_cls=lambda **_: FakeVelocityAviary(),
            drone_model=SimpleNamespace(CF2X="cf2x"),
            physics=SimpleNamespace(PYB="pyb"),
        ),
        HCAPPOTrainingConfig(
            total_timesteps=32,
            rollout_steps=16,
            minibatch_size=8,
            update_epochs=1,
            torch_num_threads=1,
            seed=17,
            network=HCAActorCriticConfig(embedding_dim=16, hidden_sizes=(8,)),
            checkpoint_path=checkpoint_path,
        ),
    )

    with pytest.raises(ValueError, match="ppo_checkpoint"):
        run_pybullet_checkpoint_evaluation(
            PyBulletCheckpointEvaluationConfig(
                checkpoint_path=checkpoint_path,
                training_settings=training_settings,
                simulation_settings=simulation_settings,
                episodes=1,
                velocity_aviary_cls=lambda **_: FakeVelocityAviary(),
                drone_model=SimpleNamespace(CF2X="cf2x"),
                physics=SimpleNamespace(PYB="pyb"),
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
