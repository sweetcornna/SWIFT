from __future__ import annotations

from pathlib import Path

import pytest

from swift.config import (
    BaselineMetadata,
    PyBulletAPFActionPriorSettings,
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


def test_load_training_settings_parses_pybullet_obstacle_randomization(tmp_path: Path) -> None:
    from swift.config import PyBulletObstacleRandomizationSettings

    config_path = tmp_path / "randomized.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pybullet_obstacle_randomization:",
                "  enabled: true",
                "  min_obstacles: 1",
                "  max_obstacles: 3",
                "  x_range: [0.12, 0.38]",
                "  y_range: [-0.30, 0.30]",
                "  z: 0.1125",
                "  radius_range: [0.04, 0.08]",
                "  endpoint_clearance: 0.02",
                "  inter_obstacle_clearance: 0.02",
                "  vehicle_radius: 0.061",
                "  require_path_blocker: true",
                "  max_sampling_attempts: 256",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_training_settings(config_path)

    assert settings.pybullet_obstacle_randomization == PyBulletObstacleRandomizationSettings(
        enabled=True,
        min_obstacles=1,
        max_obstacles=3,
        x_range=(0.12, 0.38),
        y_range=(-0.30, 0.30),
        z=0.1125,
        radius_range=(0.04, 0.08),
        endpoint_clearance=0.02,
        inter_obstacle_clearance=0.02,
        vehicle_radius=0.061,
        require_path_blocker=True,
        max_sampling_attempts=256,
    )


def test_load_training_settings_parses_pybullet_reward_and_curriculum(tmp_path: Path) -> None:
    from swift.config import (
        PyBulletCurriculumPhaseSettings,
        PyBulletCurriculumSettings,
        PyBulletRewardSettings,
    )

    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        """
pybullet_obstacle_randomization:
  enabled: true
  vehicle_radius: 0.061
pybullet_reward:
  arrival_reward: 100.0
  approach_scale: 20.0
  collision_penalty: 100.0
  timeout_penalty: 20.0
  episode_time_penalty: 1.0
  heading_smoothness_penalty: 0.05
pybullet_curriculum:
  enabled: true
  phases:
    - {name: goal_reaching, end_fraction: 0.20, min_obstacles: 0, max_obstacles: 0, require_path_blocker: false}
    - {name: single_obstacle, end_fraction: 0.40, min_obstacles: 1, max_obstacles: 1, require_path_blocker: false}
    - {name: single_blocker, end_fraction: 0.70, min_obstacles: 1, max_obstacles: 1, require_path_blocker: true}
    - {name: randomized_final, end_fraction: 1.00, min_obstacles: 1, max_obstacles: 3, require_path_blocker: true}
""".strip(),
        encoding="utf-8",
    )

    settings = load_training_settings(config_path)

    assert settings.pybullet_obstacle_randomization.vehicle_radius == pytest.approx(0.061)
    assert settings.pybullet_reward == PyBulletRewardSettings(
        arrival_reward=100.0,
        approach_scale=20.0,
        collision_penalty=100.0,
        timeout_penalty=20.0,
        episode_time_penalty=1.0,
        heading_smoothness_penalty=0.05,
    )
    assert settings.pybullet_curriculum == PyBulletCurriculumSettings(
        enabled=True,
        phases=(
            PyBulletCurriculumPhaseSettings("goal_reaching", 0.20, 0, 0, False),
            PyBulletCurriculumPhaseSettings("single_obstacle", 0.40, 1, 1, False),
            PyBulletCurriculumPhaseSettings("single_blocker", 0.70, 1, 1, True),
            PyBulletCurriculumPhaseSettings("randomized_final", 1.00, 1, 3, True),
        ),
    )
    assert settings.pybullet_curriculum.phase_for(0.0).name == "goal_reaching"
    assert settings.pybullet_curriculum.phase_for(0.20).name == "single_obstacle"
    assert settings.pybullet_curriculum.phase_for(0.70).name == "randomized_final"
    assert settings.pybullet_curriculum.phase_for(1.0).name == "randomized_final"


def test_load_training_settings_parses_pybullet_apf_action_prior(tmp_path: Path) -> None:
    config_path = tmp_path / "apf-prior.yaml"
    config_path.write_text(
        """
pybullet_apf_action_prior:
  enabled: true
  attractive_gain: 1.0
  repulsive_gain: 0.02
  influence_radius: 0.4
  max_repulsive_magnitude: 10.0
  epsilon: 0.000001
  ignore_obstacles_behind: true
  bypass_enabled: true
  bypass_lateral_offset: 0.25
  bypass_forward_margin: 0.08
  bypass_clearance: 0.16
  visibility_planner_enabled: true
  visibility_clearance: 0.18
  visibility_samples: 16
  policy_residual_scale: 0.25
""".strip(),
        encoding="utf-8",
    )

    settings = load_training_settings(config_path)

    assert settings.pybullet_apf_action_prior == PyBulletAPFActionPriorSettings(
        enabled=True,
        attractive_gain=1.0,
        repulsive_gain=0.02,
        influence_radius=0.4,
        max_repulsive_magnitude=10.0,
        epsilon=0.000001,
        ignore_obstacles_behind=True,
        bypass_enabled=True,
        bypass_lateral_offset=0.25,
        bypass_forward_margin=0.08,
        bypass_clearance=0.16,
        visibility_planner_enabled=True,
        visibility_clearance=0.18,
        visibility_samples=16,
        policy_residual_scale=0.25,
    )


@pytest.mark.parametrize(
    ("yaml_body", "message"),
    [
        ("phases: []", "phases must not be empty"),
        (
            "phases:\n"
            "    - {name: first, end_fraction: 0.7, min_obstacles: 0, max_obstacles: 0}\n"
            "    - {name: second, end_fraction: 0.6, min_obstacles: 1, max_obstacles: 1}",
            "end_fraction values must be strictly increasing",
        ),
        (
            "phases:\n"
            "    - {name: final, end_fraction: 0.9, min_obstacles: 1, max_obstacles: 1}",
            "final curriculum end_fraction must equal 1.0",
        ),
        (
            "phases:\n"
            "    - {name: empty, end_fraction: 1.0, min_obstacles: 0, max_obstacles: 0, require_path_blocker: true}",
            "path blocker requires at least one obstacle",
        ),
    ],
)
def test_load_training_settings_rejects_invalid_pybullet_curriculum(
    tmp_path: Path,
    yaml_body: str,
    message: str,
) -> None:
    path = tmp_path / "invalid-curriculum.yaml"
    path.write_text(
        "pybullet_curriculum:\n  enabled: true\n  " + yaml_body + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_training_settings(path)


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "-1.0"])
def test_load_training_settings_rejects_invalid_pybullet_reward(tmp_path: Path, value: str) -> None:
    path = tmp_path / "invalid-reward.yaml"
    path.write_text(
        f"pybullet_reward:\n  approach_scale: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approach_scale must be non-negative"):
        load_training_settings(path)


@pytest.mark.parametrize(
    ("yaml_body", "message"),
    [
        ("min_obstacles: 3\n  max_obstacles: 1", "max_obstacles must be >= min_obstacles"),
        ("x_range: [0.4, 0.1]", "x_range lower value must be <= upper value"),
        ("radius_range: [0.0, 0.1]", "radius_range values must be positive"),
        ("max_sampling_attempts: 0", "max_sampling_attempts must be positive"),
    ],
)
def test_load_training_settings_rejects_invalid_obstacle_randomization(
    tmp_path: Path,
    yaml_body: str,
    message: str,
) -> None:
    config_path = tmp_path / "invalid-randomized.yaml"
    config_path.write_text(
        "pybullet_obstacle_randomization:\n  enabled: true\n  " + yaml_body + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_training_settings(config_path)


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


def test_pybullet_randomized_training_config_uses_strict_robustness_defaults() -> None:
    settings = load_training_settings(Path("configs") / "training_pybullet_randomized.yaml")

    randomization = settings.pybullet_obstacle_randomization
    assert settings.run.variant == "ppo_mlp_pybullet_randomized"
    assert settings.run.total_timesteps == 150000
    assert settings.environment.start == pytest.approx((0.0, 0.0, 0.1125))
    assert settings.environment.goal == pytest.approx((0.5, 0.0, 0.1125))
    assert settings.environment.max_steps == 1200
    assert settings.environment.safety_margin == pytest.approx(0.1)
    assert settings.environment.obstacles == ()
    assert settings.policy.max_heading_delta == pytest.approx(0.2)
    assert settings.policy.min_speed_fraction == pytest.approx(1.0)
    assert settings.pybullet_apf_action_prior.enabled is True
    assert settings.pybullet_apf_action_prior.repulsive_gain == pytest.approx(0.002)
    assert settings.pybullet_apf_action_prior.influence_radius == pytest.approx(0.4)
    assert settings.pybullet_apf_action_prior.ignore_obstacles_behind is True
    assert settings.pybullet_apf_action_prior.bypass_enabled is False
    assert settings.pybullet_apf_action_prior.bypass_lateral_offset == pytest.approx(0.25)
    assert settings.pybullet_apf_action_prior.bypass_forward_margin == pytest.approx(0.08)
    assert settings.pybullet_apf_action_prior.bypass_clearance == pytest.approx(0.16)
    assert settings.pybullet_apf_action_prior.visibility_planner_enabled is True
    assert settings.pybullet_apf_action_prior.visibility_clearance == pytest.approx(0.18)
    assert settings.pybullet_apf_action_prior.visibility_samples == 16
    assert settings.pybullet_apf_action_prior.policy_residual_scale == pytest.approx(0.25)
    assert randomization.enabled is True
    assert (randomization.min_obstacles, randomization.max_obstacles) == (1, 3)
    assert randomization.x_range == pytest.approx((0.12, 0.38))
    assert randomization.y_range == pytest.approx((-0.30, 0.30))
    assert randomization.radius_range == pytest.approx((0.04, 0.08))
    assert randomization.vehicle_radius == pytest.approx(0.061)
    assert randomization.require_path_blocker is True
    assert randomization.max_sampling_attempts == 256
    assert settings.pybullet_reward.approach_scale == pytest.approx(20.0)
    assert settings.pybullet_reward.timeout_penalty == pytest.approx(20.0)
    assert [phase.name for phase in settings.pybullet_curriculum.phases] == [
        "goal_reaching",
        "single_obstacle",
        "single_blocker",
        "randomized_final",
    ]


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
