from __future__ import annotations

import math
from dataclasses import dataclass

from swift.core import DroneAction

OBSERVATION_SIZE = 15


@dataclass(frozen=True)
class MLPBaselinePolicyConfig:
    max_speed: float = 1.0
    max_heading_delta: float = 0.5
    max_climb_rate: float = 0.5
    obstacle_avoidance_distance: float = 2.0
    avoidance_heading_delta: float = 0.4

    def __post_init__(self) -> None:
        _set_non_negative_float(self, "max_speed", self.max_speed)
        _set_non_negative_float(self, "max_heading_delta", self.max_heading_delta)
        _set_non_negative_float(self, "max_climb_rate", self.max_climb_rate)
        _set_non_negative_float(self, "obstacle_avoidance_distance", self.obstacle_avoidance_distance)
        _set_non_negative_float(self, "avoidance_heading_delta", self.avoidance_heading_delta)


class MLPBaselinePolicy:
    """Deterministic Stage 1 policy shaped like the future MLP policy contract."""

    def __init__(self, config: MLPBaselinePolicyConfig | None = None) -> None:
        self.config = config or MLPBaselinePolicyConfig()

    def act(self, observation: tuple[float, ...]) -> DroneAction:
        if len(observation) != OBSERVATION_SIZE:
            raise ValueError(f"observation must contain exactly {OBSERVATION_SIZE} values")

        values = tuple(float(value) for value in observation)
        yaw = values[6]
        relative_goal = values[7:10]
        nearest_obstacle_relative = values[10:13]
        nearest_obstacle_radius = values[13]
        goal_distance = values[14]

        heading_delta = self._goal_heading_delta(relative_goal, yaw)
        heading_delta += self._obstacle_avoidance_delta(nearest_obstacle_relative, nearest_obstacle_radius, yaw)

        return DroneAction(
            speed=_clamp(goal_distance, 0.0, self.config.max_speed),
            heading_delta=_clamp(heading_delta, -self.config.max_heading_delta, self.config.max_heading_delta),
            climb_rate=_clamp(relative_goal[2], -self.config.max_climb_rate, self.config.max_climb_rate),
        )

    @staticmethod
    def _goal_heading_delta(relative_goal: tuple[float, float, float], yaw: float) -> float:
        goal_x, goal_y, _ = relative_goal
        if goal_x == 0.0 and goal_y == 0.0:
            return 0.0
        target_heading = math.atan2(goal_y, goal_x)
        return _normalize_angle(target_heading - yaw)

    def _obstacle_avoidance_delta(
        self,
        nearest_obstacle_relative: tuple[float, float, float],
        nearest_obstacle_radius: float,
        yaw: float,
    ) -> float:
        if nearest_obstacle_radius <= 0.0:
            return 0.0
        obstacle_distance = math.sqrt(sum(value * value for value in nearest_obstacle_relative))
        safety_distance = obstacle_distance - nearest_obstacle_radius
        if safety_distance >= self.config.obstacle_avoidance_distance:
            return 0.0

        obstacle_x, obstacle_y, _ = nearest_obstacle_relative
        obstacle_heading_delta = _normalize_angle(math.atan2(obstacle_y, obstacle_x) - yaw)
        steering_sign = -1.0 if obstacle_heading_delta >= 0.0 else 1.0
        return steering_sign * self.config.avoidance_heading_delta


def _set_non_negative_float(instance: object, name: str, value: float) -> None:
    numeric_value = float(value)
    if numeric_value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    object.__setattr__(instance, name, numeric_value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
