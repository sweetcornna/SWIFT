from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from swift.core import (
    DroneAction,
    DroneState,
    EpisodeMetrics,
    ObstacleState,
    RewardBreakdown,
    Vector3,
    compute_path_length,
    compute_path_smoothness,
)


def _vector3(name: str, value: Sequence[float]) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (float(value[0]), float(value[1]), float(value[2]))


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class SimpleAvoidanceSettings:
    start: Vector3 = (0.0, 0.0, 0.0)
    goal: Vector3 = (10.0, 0.0, 0.0)
    obstacles: tuple[ObstacleState, ...] = ()
    max_steps: int = 100
    goal_radius: float = 0.5
    safety_margin: float = 0.25
    time_delta: float = 1.0
    max_speed: float = 1.0
    max_climb_rate: float = 0.5
    world_bounds: tuple[Vector3, Vector3] = (
        (-100.0, -100.0, -100.0),
        (100.0, 100.0, 100.0),
    )

    def __post_init__(self) -> None:
        lower, upper = self.world_bounds
        lower_bound = _vector3("world_bounds lower", lower)
        upper_bound = _vector3("world_bounds upper", upper)
        if any(lower_bound[index] > upper_bound[index] for index in range(3)):
            raise ValueError("world_bounds lower values must be <= upper values")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.goal_radius <= 0.0:
            raise ValueError("goal_radius must be positive")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")
        if self.time_delta <= 0.0:
            raise ValueError("time_delta must be positive")
        if self.max_speed < 0.0:
            raise ValueError("max_speed must be non-negative")
        if self.max_climb_rate < 0.0:
            raise ValueError("max_climb_rate must be non-negative")

        object.__setattr__(self, "start", _vector3("start", self.start))
        object.__setattr__(self, "goal", _vector3("goal", self.goal))
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "goal_radius", float(self.goal_radius))
        object.__setattr__(self, "safety_margin", float(self.safety_margin))
        object.__setattr__(self, "time_delta", float(self.time_delta))
        object.__setattr__(self, "max_speed", float(self.max_speed))
        object.__setattr__(self, "max_climb_rate", float(self.max_climb_rate))
        object.__setattr__(self, "world_bounds", (lower_bound, upper_bound))

    def replace(self, **changes: Any) -> SimpleAvoidanceSettings:
        return replace(self, **changes)


class SimpleAvoidanceEnv:
    def __init__(self, settings: SimpleAvoidanceSettings | None = None) -> None:
        self.settings = settings or SimpleAvoidanceSettings()
        self._state = DroneState(position=self.settings.start, velocity=(0.0, 0.0, 0.0), yaw=0.0)
        self._obstacles = self.settings.obstacles
        self._steps = 0
        self._path: list[Vector3] = [self.settings.start]
        self._minimum_safety_distance = self._compute_minimum_safety_distance()

    @property
    def observation_shape(self) -> tuple[int]:
        return (15,)

    @property
    def action_shape(self) -> tuple[int]:
        return (3,)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        del seed, options
        self._state = DroneState(position=self.settings.start, velocity=(0.0, 0.0, 0.0), yaw=0.0)
        self._obstacles = self.settings.obstacles
        self._steps = 0
        self._path = [self.settings.start]
        self._minimum_safety_distance = self._compute_minimum_safety_distance()
        return self._observation(), self._info(
            reward_breakdown=RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
            reached_goal=False,
            collided=False,
            timed_out=False,
        )

    def step(
        self,
        action: DroneAction | Sequence[float],
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        previous_goal_distance = self._goal_distance()
        drone_action = self._coerce_action(action)
        speed = _clamp(float(drone_action.speed), 0.0, self.settings.max_speed)
        climb_rate = _clamp(
            float(drone_action.climb_rate),
            -self.settings.max_climb_rate,
            self.settings.max_climb_rate,
        )
        yaw = self._state.yaw + float(drone_action.heading_delta)
        commanded_velocity = (
            speed * math.cos(yaw),
            speed * math.sin(yaw),
            climb_rate,
        )
        unclamped_position = tuple(
            self._state.position[index] + commanded_velocity[index] * self.settings.time_delta
            for index in range(3)
        )
        position = self._clamp_position(unclamped_position)
        velocity = tuple(
            (position[index] - self._state.position[index]) / self.settings.time_delta
            for index in range(3)
        )

        self._state = DroneState(position=position, velocity=velocity, yaw=yaw)
        self._obstacles = self._move_obstacles()
        self._steps += 1
        self._path.append(self._state.position)
        self._minimum_safety_distance = min(
            self._minimum_safety_distance,
            self._compute_minimum_safety_distance(),
        )

        reached_goal = self._goal_distance() <= self.settings.goal_radius
        collided = self._has_collision()
        terminated = reached_goal or collided
        timed_out = not terminated and self._steps >= self.settings.max_steps
        truncated = timed_out

        reward_breakdown = self._reward_breakdown(
            previous_goal_distance=previous_goal_distance,
            reached_goal=reached_goal,
            collided=collided,
            heading_delta=float(drone_action.heading_delta),
        )
        info = self._info(
            reward_breakdown=reward_breakdown,
            reached_goal=reached_goal,
            collided=collided,
            timed_out=timed_out,
        )
        if terminated or truncated:
            info["episode_metrics"] = EpisodeMetrics(
                reached_goal=reached_goal,
                collided=collided,
                timed_out=timed_out,
                path_length=compute_path_length(tuple(self._path)),
                path_smoothness=compute_path_smoothness(tuple(self._path)),
                minimum_safety_distance=self._minimum_safety_distance,
                steps=self._steps,
            )

        return self._observation(), reward_breakdown.total, terminated, truncated, info

    def _coerce_action(self, action: DroneAction | Sequence[float]) -> DroneAction:
        if isinstance(action, DroneAction):
            return action
        if len(action) != 3:
            raise ValueError("action must contain exactly 3 values")
        return DroneAction(speed=float(action[0]), heading_delta=float(action[1]), climb_rate=float(action[2]))

    def _clamp_position(self, position: Sequence[float]) -> Vector3:
        lower, upper = self.settings.world_bounds
        return tuple(_clamp(float(position[index]), lower[index], upper[index]) for index in range(3))

    def _move_obstacles(self) -> tuple[ObstacleState, ...]:
        moved = []
        for obstacle in self._obstacles:
            moved.append(
                ObstacleState(
                    position=self._clamp_position(
                        tuple(
                            obstacle.position[index] + obstacle.velocity[index] * self.settings.time_delta
                            for index in range(3)
                        )
                    ),
                    radius=obstacle.radius,
                    velocity=obstacle.velocity,
                )
            )
        return tuple(moved)

    def _nearest_obstacle(self) -> ObstacleState | None:
        if not self._obstacles:
            return None
        return min(self._obstacles, key=lambda obstacle: _distance(self._state.position, obstacle.position))

    def _observation(self) -> tuple[float, ...]:
        relative_goal = tuple(self.settings.goal[index] - self._state.position[index] for index in range(3))
        nearest_obstacle = self._nearest_obstacle()
        if nearest_obstacle is None:
            relative_obstacle = (0.0, 0.0, 0.0)
            obstacle_radius = 0.0
        else:
            relative_obstacle = tuple(
                nearest_obstacle.position[index] - self._state.position[index] for index in range(3)
            )
            obstacle_radius = float(nearest_obstacle.radius)

        return (
            *self._state.position,
            *self._state.velocity,
            self._state.yaw,
            *relative_goal,
            *relative_obstacle,
            obstacle_radius,
            self._goal_distance(),
        )

    def _info(
        self,
        *,
        reward_breakdown: RewardBreakdown,
        reached_goal: bool,
        collided: bool,
        timed_out: bool,
    ) -> dict[str, Any]:
        return {
            "drone_state": self._state,
            "obstacles": self._obstacles,
            "reward_breakdown": reward_breakdown,
            "reached_goal": reached_goal,
            "collided": collided,
            "timed_out": timed_out,
        }

    def _goal_distance(self) -> float:
        return _distance(self._state.position, self.settings.goal)

    def _has_collision(self) -> bool:
        return any(
            _distance(self._state.position, obstacle.position)
            <= obstacle.radius + self.settings.safety_margin
            for obstacle in self._obstacles
        )

    def _compute_minimum_safety_distance(self) -> float:
        if not self._obstacles:
            return 0.0
        return min(
            _distance(self._state.position, obstacle.position) - obstacle.radius
            for obstacle in self._obstacles
        )

    def _reward_breakdown(
        self,
        *,
        previous_goal_distance: float,
        reached_goal: bool,
        collided: bool,
        heading_delta: float,
    ) -> RewardBreakdown:
        approach = previous_goal_distance - self._goal_distance()
        return RewardBreakdown(
            arrive=100.0 if reached_goal else 0.0,
            approach=approach,
            obstacle=-100.0 if collided else 0.0,
            smoothness=-0.05 * abs(heading_delta),
            timeliness=-1.0,
        )
