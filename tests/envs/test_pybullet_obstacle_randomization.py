from __future__ import annotations

import math

import pytest

from swift.envs import SimpleAvoidanceSettings


def test_randomized_obstacles_are_deterministic_and_satisfy_constraints() -> None:
    from swift.config import PyBulletObstacleRandomizationSettings
    from swift.envs import sample_pybullet_obstacles

    randomization = PyBulletObstacleRandomizationSettings(
        enabled=True,
        min_obstacles=3,
        max_obstacles=3,
        x_range=(0.12, 0.38),
        y_range=(-0.30, 0.30),
        z=0.1125,
        radius_range=(0.04, 0.08),
        endpoint_clearance=0.02,
        inter_obstacle_clearance=0.02,
        require_path_blocker=True,
        max_sampling_attempts=256,
    )
    environment = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.1125),
        goal=(0.5, 0.0, 0.1125),
        safety_margin=0.1,
    )

    first = sample_pybullet_obstacles(randomization, environment, seed=1234)
    second = sample_pybullet_obstacles(randomization, environment, seed=1234)

    assert first == second
    assert len(first) == 3
    for obstacle in first:
        x, y, z = obstacle.position
        assert 0.12 <= x <= 0.38
        assert -0.30 <= y <= 0.30
        assert z == pytest.approx(0.1125)
        assert 0.04 <= obstacle.radius <= 0.08
        required_endpoint_distance = obstacle.radius + environment.safety_margin + 0.02
        assert math.dist(obstacle.position, environment.start) >= required_endpoint_distance
        assert math.dist(obstacle.position, environment.goal) >= required_endpoint_distance
    for index, obstacle in enumerate(first):
        for other in first[index + 1 :]:
            assert math.dist(obstacle.position, other.position) >= (
                obstacle.radius + other.radius + randomization.inter_obstacle_clearance
            )
    assert any(
        environment.start[0] <= obstacle.position[0] <= environment.goal[0]
        and abs(obstacle.position[1]) <= obstacle.radius + environment.safety_margin
        for obstacle in first
    )


def test_randomized_obstacles_change_with_scenario_seed() -> None:
    from swift.config import PyBulletObstacleRandomizationSettings
    from swift.envs import sample_pybullet_obstacles

    randomization = PyBulletObstacleRandomizationSettings(enabled=True)
    environment = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.1125),
        goal=(0.5, 0.0, 0.1125),
        safety_margin=0.1,
    )

    assert sample_pybullet_obstacles(randomization, environment, seed=1) != sample_pybullet_obstacles(
        randomization,
        environment,
        seed=2,
    )


def test_randomized_obstacles_report_impossible_constraints() -> None:
    from swift.config import PyBulletObstacleRandomizationSettings
    from swift.envs import sample_pybullet_obstacles

    randomization = PyBulletObstacleRandomizationSettings(
        enabled=True,
        min_obstacles=1,
        max_obstacles=1,
        x_range=(0.0, 0.0),
        y_range=(0.0, 0.0),
        z=0.0,
        radius_range=(0.08, 0.08),
        endpoint_clearance=0.02,
        max_sampling_attempts=1,
    )
    environment = SimpleAvoidanceSettings(start=(0.0, 0.0, 0.0), goal=(0.5, 0.0, 0.0), safety_margin=0.1)

    with pytest.raises(RuntimeError, match="seed=9"):
        sample_pybullet_obstacles(randomization, environment, seed=9)
