import math

import pytest

from swift.rl.visibility_planner import (
    VisibilityPlannerConfig,
    visibility_waypoint_from_observation,
)


def _observation(
    *,
    relative_goal: tuple[float, float, float] = (0.5, 0.0, 0.0),
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    slots: int = 3,
) -> tuple[float, ...]:
    nearest = min(obstacles, key=lambda item: math.dist(item[:3], (0.0, 0.0, 0.0))) if obstacles else None
    prefix = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *relative_goal,
        *(nearest[:3] if nearest is not None else (0.0, 0.0, 0.0)),
        nearest[3] if nearest is not None else 0.0,
        math.dist(relative_goal, (0.0, 0.0, 0.0)),
    )
    padded = (*obstacles, *((0.0, 0.0, 0.0, 0.0),) * (slots - len(obstacles)))
    return (*prefix, *(value for obstacle in padded for value in obstacle))


def _segment_distance_to_point(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(start, point)
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    projection = min(max(projection, 0.0), 1.0)
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.dist(closest, point)


def test_visibility_planner_returns_goal_when_direct_segment_is_clear() -> None:
    observation = _observation(obstacles=((0.25, 0.35, 0.0, 0.05),))

    waypoint = visibility_waypoint_from_observation(observation)

    assert waypoint == pytest.approx((0.5, 0.0, 0.0))


def test_visibility_planner_routes_laterally_around_centered_blocker() -> None:
    obstacle = (0.25, 0.0, 0.0, 0.05)
    config = VisibilityPlannerConfig(clearance=0.18, samples=16)

    waypoint = visibility_waypoint_from_observation(
        _observation(obstacles=(obstacle,)),
        config,
    )

    assert waypoint[0] > 0.0
    assert abs(waypoint[1]) > 0.0
    assert (
        _segment_distance_to_point((0.0, 0.0), waypoint[:2], obstacle[:2]) + 1e-9
        >= obstacle[3] + config.clearance
    )


def test_visibility_planner_routes_outside_overlapping_obstacle_cluster() -> None:
    obstacles = (
        (0.25, -0.12, 0.0, 0.04),
        (0.25, 0.12, 0.0, 0.04),
    )
    config = VisibilityPlannerConfig(clearance=0.18, samples=16)

    waypoint = visibility_waypoint_from_observation(
        _observation(obstacles=obstacles),
        config,
    )

    assert abs(waypoint[1]) > 0.12
    for obstacle in obstacles:
        assert (
            _segment_distance_to_point((0.0, 0.0), waypoint[:2], obstacle[:2]) + 1e-9
            >= obstacle[3] + config.clearance
        )


def test_visibility_planner_is_deterministic_for_symmetric_layout() -> None:
    observation = _observation(obstacles=((0.25, 0.0, 0.0, 0.05),))

    first = visibility_waypoint_from_observation(observation)
    second = visibility_waypoint_from_observation(observation)

    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"clearance": 0.0}, "clearance"),
        ({"samples": 7}, "samples"),
    ],
)
def test_visibility_planner_rejects_invalid_config(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VisibilityPlannerConfig(**kwargs)
