from dataclasses import asdict
import json
import math

import pytest

from swift.core import DroneAction, EpisodeMetrics, ObstacleState, RewardBreakdown
from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings


def _settings(**changes):
    return SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 0.0, 0.0),
        obstacles=(),
        max_steps=10,
        goal_radius=0.25,
        safety_margin=0.1,
        time_delta=1.0,
        max_speed=2.0,
        max_climb_rate=1.0,
        world_bounds=((-20.0, -20.0, -5.0), (20.0, 20.0, 10.0)),
    ).replace(**changes)


def test_settings_are_frozen_and_replaceable():
    settings = _settings(max_steps=8)
    updated = settings.replace(max_steps=3)

    assert settings.max_steps == 8
    assert updated.max_steps == 3
    with pytest.raises(Exception):
        settings.max_steps = 9


def test_reset_returns_observation_shape_and_state_info():
    obstacle = ObstacleState(position=(2.0, 2.0, 3.0), radius=0.5)
    env = SimpleAvoidanceEnv(
        _settings(start=(1.0, 2.0, 3.0), goal=(4.0, 6.0, 3.0), obstacles=(obstacle,))
    )

    observation, info = env.reset(seed=123, options={"unused": True})

    assert env.observation_shape == (15,)
    assert env.action_shape == (3,)
    assert observation == pytest.approx(
        (
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            4.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.5,
            5.0,
        )
    )
    assert info["drone_state"].position == (1.0, 2.0, 3.0)
    assert info["obstacles"] == (obstacle,)
    assert info["reached_goal"] is False
    assert info["collided"] is False
    assert info["timed_out"] is False
    assert "episode_metrics" not in info


def test_drone_action_and_sequence_actions_update_deterministically():
    env = SimpleAvoidanceEnv(_settings(goal=(50.0, 50.0, 10.0)))
    env.reset()

    observation, reward, terminated, truncated, info = env.step(
        DroneAction(speed=2.0, heading_delta=0.0, climb_rate=0.0)
    )

    assert observation[:7] == pytest.approx((2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0))
    assert reward == pytest.approx(info["reward_breakdown"].total)
    assert isinstance(info["reward_breakdown"], RewardBreakdown)
    assert terminated is False
    assert truncated is False

    observation, reward, terminated, truncated, info = env.step((10.0, math.pi / 2.0, 5.0))

    assert observation[:7] == pytest.approx(
        (2.0, 2.0, 1.0, 0.0, 2.0, 1.0, math.pi / 2.0),
        abs=1e-9,
    )
    assert reward == pytest.approx(info["reward_breakdown"].total)
    assert terminated is False
    assert truncated is False


def test_goal_reaching_terminates_with_episode_metrics():
    env = SimpleAvoidanceEnv(_settings(goal=(1.0, 0.0, 0.0), max_speed=1.0))
    env.reset()

    _, _, terminated, truncated, info = env.step((1.0, 0.0, 0.0))

    assert terminated is True
    assert truncated is False
    assert info["reached_goal"] is True
    assert info["collided"] is False
    assert isinstance(info["episode_metrics"], EpisodeMetrics)
    assert info["episode_metrics"].success is True
    assert info["episode_metrics"].steps == 1


def test_collision_terminates_with_episode_metrics():
    obstacle = ObstacleState(position=(1.0, 0.0, 0.0), radius=0.5)
    env = SimpleAvoidanceEnv(_settings(obstacles=(obstacle,), max_speed=1.0))
    env.reset()

    _, _, terminated, truncated, info = env.step((1.0, 0.0, 0.0))

    assert terminated is True
    assert truncated is False
    assert info["reached_goal"] is False
    assert info["collided"] is True
    assert info["episode_metrics"].collided is True
    assert info["episode_metrics"].success is False


def test_timeout_truncates_with_episode_metrics():
    env = SimpleAvoidanceEnv(_settings(goal=(50.0, 0.0, 0.0), max_steps=1))
    env.reset()

    _, _, terminated, truncated, info = env.step((0.0, 0.0, 0.0))

    assert terminated is False
    assert truncated is True
    assert info["timed_out"] is True
    assert info["episode_metrics"].timed_out is True
    assert info["episode_metrics"].steps == 1


def test_dynamic_obstacles_move_before_next_observation():
    obstacle = ObstacleState(position=(5.0, 5.0, 0.0), radius=0.5, velocity=(0.5, -1.0, 0.25))
    env = SimpleAvoidanceEnv(_settings(goal=(50.0, 50.0, 0.0), obstacles=(obstacle,), time_delta=2.0))
    env.reset()

    observation, _, terminated, truncated, info = env.step((0.0, 0.0, 0.0))

    assert info["obstacles"][0].position == pytest.approx((6.0, 3.0, 0.5))
    assert observation[10:14] == pytest.approx((6.0, 3.0, 0.5, 0.5))
    assert terminated is False
    assert truncated is False


def test_no_obstacle_terminal_metrics_are_json_safe():
    env = SimpleAvoidanceEnv(_settings(goal=(1.0, 0.0, 0.0), max_speed=1.0, obstacles=()))
    env.reset()

    _, _, terminated, truncated, info = env.step((1.0, 0.0, 0.0))

    assert terminated is True
    assert truncated is False
    assert info["episode_metrics"].minimum_safety_distance == 0.0
    json.dumps(asdict(info["episode_metrics"]), allow_nan=False)
