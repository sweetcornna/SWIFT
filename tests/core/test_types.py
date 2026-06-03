import pytest

from swift.core import DroneAction, DroneState, EpisodeMetrics, ObstacleState, RewardBreakdown


def test_drone_state_is_three_dimensional():
    state = DroneState(position=(1.0, 2.0, 3.0), velocity=(0.1, 0.2, 0.3), yaw=0.5)

    assert state.position == (1.0, 2.0, 3.0)
    assert state.velocity == (0.1, 0.2, 0.3)


def test_drone_state_rejects_invalid_vector_length():
    with pytest.raises(ValueError, match="position must contain exactly 3 values"):
        DroneState(position=(1.0, 2.0), velocity=(0.0, 0.0, 0.0), yaw=0.0)


def test_reward_breakdown_total_sums_terms():
    reward = RewardBreakdown(
        arrive=10.0,
        approach=1.5,
        obstacle=-2.0,
        smoothness=-0.5,
        timeliness=-1.0,
    )

    assert reward.total == 8.0


def test_episode_metrics_success_flag_is_explicit():
    metrics = EpisodeMetrics(
        reached_goal=True,
        collided=False,
        timed_out=False,
        path_length=12.0,
        path_smoothness=0.2,
        minimum_safety_distance=1.5,
        steps=120,
    )

    assert metrics.success is True
