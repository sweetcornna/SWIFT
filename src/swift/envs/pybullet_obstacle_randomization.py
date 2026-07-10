from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from swift.core import ObstacleState

if TYPE_CHECKING:
    from swift.config import PyBulletObstacleRandomizationSettings
    from swift.envs.simple_avoidance import SimpleAvoidanceSettings


def sample_pybullet_obstacles(
    randomization: PyBulletObstacleRandomizationSettings,
    environment: SimpleAvoidanceSettings,
    *,
    seed: int,
    phase_name: str = "final",
) -> tuple[ObstacleState, ...]:
    if not randomization.enabled:
        return environment.obstacles

    rng = random.Random(int(seed))
    for _ in range(randomization.max_sampling_attempts):
        count = rng.randint(randomization.min_obstacles, randomization.max_obstacles)
        obstacles: list[ObstacleState] = []
        for _ in range(count):
            obstacle = ObstacleState(
                position=(
                    rng.uniform(*randomization.x_range),
                    rng.uniform(*randomization.y_range),
                    randomization.z,
                ),
                radius=rng.uniform(*randomization.radius_range),
            )
            if not _valid_candidate(obstacle, obstacles, randomization, environment):
                break
            obstacles.append(obstacle)
        if len(obstacles) != count:
            continue
        if randomization.require_path_blocker and not any(
            _blocks_direct_path(obstacle, randomization, environment) for obstacle in obstacles
        ):
            continue
        return tuple(obstacles)

    required_endpoint_clearance = (
        environment.safety_margin + randomization.vehicle_radius + randomization.endpoint_clearance
    )
    raise RuntimeError(
        "Unable to sample a feasible PyBullet obstacle layout "
        f"for seed={int(seed)} phase={str(phase_name)} "
        f"obstacle_count={randomization.min_obstacles}..{randomization.max_obstacles} "
        f"required_endpoint_clearance={required_endpoint_clearance:.3f} "
        f"after {randomization.max_sampling_attempts} attempts"
    )


def _valid_candidate(
    candidate: ObstacleState,
    existing: list[ObstacleState],
    randomization: PyBulletObstacleRandomizationSettings,
    environment: SimpleAvoidanceSettings,
) -> bool:
    endpoint_distance = (
        candidate.radius
        + randomization.vehicle_radius
        + environment.safety_margin
        + randomization.endpoint_clearance
    )
    if math.dist(candidate.position, environment.start) < endpoint_distance:
        return False
    if math.dist(candidate.position, environment.goal) < endpoint_distance:
        return False
    return all(
        math.dist(candidate.position, obstacle.position)
        >= candidate.radius + obstacle.radius + randomization.inter_obstacle_clearance
        for obstacle in existing
    )


def _blocks_direct_path(
    obstacle: ObstacleState,
    randomization: PyBulletObstacleRandomizationSettings,
    environment: SimpleAvoidanceSettings,
) -> bool:
    start = environment.start
    goal = environment.goal
    segment = tuple(goal[index] - start[index] for index in range(3))
    segment_length_squared = sum(component * component for component in segment)
    if segment_length_squared <= 0.0:
        return False
    offset = tuple(obstacle.position[index] - start[index] for index in range(3))
    projection = sum(offset[index] * segment[index] for index in range(3)) / segment_length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = tuple(start[index] + projection * segment[index] for index in range(3))
    return math.dist(obstacle.position, nearest) <= (
        obstacle.radius + randomization.vehicle_radius + environment.safety_margin
    )
