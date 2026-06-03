import pytest

from swift.envs import SimpleAvoidanceEnv
from swift.rl import MLPBaselinePolicy, MLPBaselinePolicyConfig


def _observation(
    *,
    yaw: float = 0.0,
    relative_goal: tuple[float, float, float] = (10.0, 10.0, 5.0),
    obstacle_relative: tuple[float, float, float] = (100.0, 100.0, 0.0),
    obstacle_radius: float = 0.5,
    goal_distance: float = 15.0,
) -> tuple[float, ...]:
    return (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        yaw,
        *relative_goal,
        *obstacle_relative,
        obstacle_radius,
        goal_distance,
    )


def test_policy_action_respects_configured_bounds():
    policy = MLPBaselinePolicy(
        MLPBaselinePolicyConfig(
            max_speed=2.0,
            max_heading_delta=0.3,
            max_climb_rate=0.2,
            obstacle_avoidance_distance=2.0,
            avoidance_heading_delta=0.5,
        )
    )

    action = policy.act(_observation(relative_goal=(10.0, 10.0, 5.0)))

    assert 0.0 <= action.speed <= 2.0
    assert -0.3 <= action.heading_delta <= 0.3
    assert -0.2 <= action.climb_rate <= 0.2


def test_policy_steers_away_from_close_obstacle():
    policy = MLPBaselinePolicy(
        MLPBaselinePolicyConfig(
            max_speed=1.0,
            max_heading_delta=1.0,
            max_climb_rate=1.0,
            obstacle_avoidance_distance=2.0,
            avoidance_heading_delta=0.4,
        )
    )

    clear_action = policy.act(
        _observation(
            relative_goal=(10.0, 0.0, 0.0),
            obstacle_relative=(10.0, 10.0, 0.0),
        )
    )
    obstacle_left_action = policy.act(
        _observation(
            relative_goal=(10.0, 0.0, 0.0),
            obstacle_relative=(1.0, 0.5, 0.0),
        )
    )

    assert clear_action.heading_delta == pytest.approx(0.0)
    assert obstacle_left_action.heading_delta < clear_action.heading_delta


def test_policy_does_not_steer_when_environment_reports_no_obstacles():
    observation, _ = SimpleAvoidanceEnv().reset()
    policy = MLPBaselinePolicy()

    action = policy.act(observation)

    assert action.heading_delta == pytest.approx(0.0)


def test_policy_rejects_invalid_observation_length():
    policy = MLPBaselinePolicy()

    with pytest.raises(ValueError, match="15"):
        policy.act((0.0,) * 14)
