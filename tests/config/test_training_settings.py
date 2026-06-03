from __future__ import annotations

from pathlib import Path

import pytest

from swift.config import (
    BaselineMetadata,
    TrainingRunSettings,
    TrainingSettings,
    load_training_settings,
)
from swift.core import ObstacleState
from swift.envs import SimpleAvoidanceSettings
from swift.experiments.artifacts import ExperimentArtifactConfig
from swift.rl import MLPBaselinePolicyConfig, PPOConfig


def _write_training_yaml(path: Path, *, ppo_overrides: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                "ppo:",
                "  rollout_steps: 128",
                "  minibatch_size: 32",
                "  update_epochs: 4",
                "  clip_range: 0.15",
                "  gamma: 0.98",
                "  gae_lambda: 0.9",
                "  entropy_coef: 0.02",
                "  value_loss_coef: 0.4",
                "  max_grad_norm: 0.6",
                *ppo_overrides.splitlines(),
                "policy:",
                "  max_speed: 2.5",
                "  max_heading_delta: 0.7",
                "  min_speed_fraction: 0.35",
                "  max_climb_rate: 0.8",
                "  obstacle_avoidance_distance: 3.5",
                "  avoidance_heading_delta: 0.55",
                "environment:",
                "  start: [0, 0, 1]",
                "  goal: [8, 1, 1]",
                "  obstacles:",
                "    - position: [2, 0, 1]",
                "      radius: 0.75",
                "      velocity: [0.1, 0, 0]",
                "    - position: [4, 1, 1]",
                "      radius: 0.5",
                "  max_steps: 80",
                "  goal_radius: 0.25",
                "  safety_margin: 0.2",
                "  time_delta: 0.5",
                "  max_speed: 1.5",
                "  max_climb_rate: 0.4",
                "  world_bounds:",
                "    - [-10, -10, 0]",
                "    - [10, 10, 5]",
                "run:",
                "  stage: stage1",
                "  variant: ppo_mlp_contract",
                "  seed: 42",
                "  episodes: 6",
                "  total_timesteps: 256",
                "artifact:",
                "  root: artifacts/out",
                "  episode_logs: artifacts/episodes",
                "  experiment_reports: artifacts/reports",
                "  checkpoints: artifacts/checkpoints",
                "baseline:",
                "  model: MLP",
                "  purpose: deterministic baseline metadata",
            ]
        ),
        encoding="utf-8",
    )


def test_load_training_settings_parses_typed_sections_and_obstacles(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    _write_training_yaml(config_path)

    settings = load_training_settings(config_path)

    assert settings == TrainingSettings(
        ppo=PPOConfig(
            rollout_steps=128,
            minibatch_size=32,
            update_epochs=4,
            clip_range=0.15,
            gamma=0.98,
            gae_lambda=0.9,
            entropy_coef=0.02,
            value_loss_coef=0.4,
            max_grad_norm=0.6,
        ),
        policy=MLPBaselinePolicyConfig(
            max_speed=2.5,
            max_heading_delta=0.7,
            min_speed_fraction=0.35,
            max_climb_rate=0.8,
            obstacle_avoidance_distance=3.5,
            avoidance_heading_delta=0.55,
        ),
        environment=SimpleAvoidanceSettings(
            start=(0.0, 0.0, 1.0),
            goal=(8.0, 1.0, 1.0),
            obstacles=(
                ObstacleState(position=(2.0, 0.0, 1.0), radius=0.75, velocity=(0.1, 0.0, 0.0)),
                ObstacleState(position=(4.0, 1.0, 1.0), radius=0.5),
            ),
            max_steps=80,
            goal_radius=0.25,
            safety_margin=0.2,
            time_delta=0.5,
            max_speed=1.5,
            max_climb_rate=0.4,
            world_bounds=((-10.0, -10.0, 0.0), (10.0, 10.0, 5.0)),
        ),
        run=TrainingRunSettings(
            stage="stage1",
            variant="ppo_mlp_contract",
            seed=42,
            episodes=6,
            total_timesteps=256,
        ),
        artifact=ExperimentArtifactConfig(
            root=Path("artifacts/out"),
            episode_logs=Path("artifacts/episodes"),
            experiment_reports=Path("artifacts/reports"),
            checkpoints=Path("artifacts/checkpoints"),
        ),
        baseline=BaselineMetadata(model="MLP", purpose="deterministic baseline metadata"),
    )


def test_current_training_config_loads_with_defaults_for_missing_sections() -> None:
    settings = load_training_settings(Path("configs") / "training.yaml")

    assert settings.ppo.rollout_steps == 2048
    assert settings.ppo.gamma == pytest.approx(0.99)
    assert settings.policy == MLPBaselinePolicyConfig()
    assert settings.environment == SimpleAvoidanceSettings(goal=(4.0, 0.0, 0.0), max_steps=8)
    assert settings.run == TrainingRunSettings(
        stage="stage1",
        variant="ppo_mlp",
        seed=0,
        episodes=10,
        total_timesteps=128,
    )
    assert settings.artifact == ExperimentArtifactConfig(
        root=Path("outputs"),
        episode_logs=Path("outputs/episodes"),
        experiment_reports=Path("outputs/reports"),
        checkpoints=Path("checkpoints/stage1"),
    )
    assert settings.baseline == BaselineMetadata(
        model="MLP",
        purpose="stable demonstrable baseline before HCA and APF fusion",
    )


def test_pybullet_probe_training_config_uses_reachable_physics_horizon() -> None:
    settings = load_training_settings(Path("configs") / "training_pybullet_probe.yaml")

    assert settings.run.variant == "ppo_mlp_pybullet_probe"
    assert settings.run.total_timesteps >= 8192
    assert settings.environment.goal == pytest.approx((0.5, 0.0, 0.1125))
    assert settings.environment.max_steps >= 600
    assert settings.environment.max_climb_rate == pytest.approx(0.0)
    assert settings.environment.obstacles == ()
    assert 0.15 <= settings.policy.max_heading_delta <= 0.25
    assert 0.35 <= settings.policy.min_speed_fraction <= 0.5


def test_pybullet_obstacle_training_config_uses_explicit_swift_obstacles() -> None:
    settings = load_training_settings(Path("configs") / "training_pybullet_obstacles.yaml")

    assert settings.run.variant == "ppo_mlp_pybullet_obstacles"
    assert settings.run.total_timesteps >= 16384
    assert settings.environment.goal == pytest.approx((0.5, 0.0, 0.1125))
    assert settings.environment.max_steps >= 600
    assert settings.environment.max_climb_rate == pytest.approx(0.0)
    assert 0.15 <= settings.policy.max_heading_delta <= 0.25
    assert 0.35 <= settings.policy.min_speed_fraction <= 0.5
    assert settings.environment.obstacles
    assert all(obstacle.radius < 2.0 for obstacle in settings.environment.obstacles)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("  rollout_steps: 0", "rollout_steps must be positive"),
        ("  gamma: 1.01", "gamma must be greater than 0 and <= 1"),
        ("  gae_lambda: 0", "gae_lambda must be greater than 0 and <= 1"),
    ],
)
def test_load_training_settings_rejects_invalid_ppo_values(
    tmp_path: Path,
    override: str,
    message: str,
) -> None:
    config_path = tmp_path / "training.yaml"
    _write_training_yaml(config_path, ppo_overrides=override)

    with pytest.raises(ValueError, match=message):
        load_training_settings(config_path)


def test_load_training_settings_rejects_non_mapping_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text("ppo: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ppo must be a mapping"):
        load_training_settings(config_path)


def test_load_training_settings_accepts_zero_ppo_loss_coefficients(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    _write_training_yaml(
        config_path,
        ppo_overrides="\n".join(
            [
                "  entropy_coef: 0.0",
                "  value_loss_coef: 0.0",
            ]
        ),
    )

    settings = load_training_settings(config_path)

    assert settings.ppo.entropy_coef == 0.0
    assert settings.ppo.value_loss_coef == 0.0
