from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from swift.rl.visibility_planner import VisibilityPlannerConfig, visibility_waypoint_from_observation


@dataclass(frozen=True)
class APFConfig:
    attractive_gain: float = 1.0
    repulsive_gain: float = 1.0
    influence_radius: float = 2.0
    max_repulsive_magnitude: float = 10.0
    epsilon: float = 1e-6
    ignore_obstacles_behind: bool = False
    bypass_enabled: bool = False
    bypass_lateral_offset: float = 0.25
    bypass_forward_margin: float = 0.08
    bypass_clearance: float = 0.16
    visibility_planner_enabled: bool = False
    visibility_clearance: float = 0.18
    visibility_samples: int = 16
    policy_residual_scale: float = 1.0

    def __post_init__(self) -> None:
        _set_non_negative_float(self, "attractive_gain", self.attractive_gain)
        _set_non_negative_float(self, "repulsive_gain", self.repulsive_gain)
        _set_positive_float(self, "influence_radius", self.influence_radius)
        _set_positive_float(self, "max_repulsive_magnitude", self.max_repulsive_magnitude)
        _set_positive_float(self, "epsilon", self.epsilon)
        object.__setattr__(self, "ignore_obstacles_behind", bool(self.ignore_obstacles_behind))
        object.__setattr__(self, "bypass_enabled", bool(self.bypass_enabled))
        _set_positive_float(self, "bypass_lateral_offset", self.bypass_lateral_offset)
        _set_non_negative_float(self, "bypass_forward_margin", self.bypass_forward_margin)
        _set_non_negative_float(self, "bypass_clearance", self.bypass_clearance)
        object.__setattr__(self, "visibility_planner_enabled", bool(self.visibility_planner_enabled))
        _set_positive_float(self, "visibility_clearance", self.visibility_clearance)
        visibility_samples = int(self.visibility_samples)
        if visibility_samples < 8:
            raise ValueError("visibility_samples must be at least 8")
        object.__setattr__(self, "visibility_samples", visibility_samples)
        policy_residual_scale = float(self.policy_residual_scale)
        if not math.isfinite(policy_residual_scale) or not 0.0 <= policy_residual_scale <= 1.0:
            raise ValueError("policy_residual_scale must be between 0 and 1")
        object.__setattr__(self, "policy_residual_scale", policy_residual_scale)


@dataclass(frozen=True)
class APFFeatureVector:
    attractive: tuple[float, float, float]
    repulsive: tuple[float, float, float]
    combined: tuple[float, float, float]

    def as_tuple(self) -> tuple[float, ...]:
        return (*self.attractive, *self.repulsive, *self.combined)


def apf_features_from_observation(
    observation: Sequence[float],
    config: APFConfig | None = None,
) -> APFFeatureVector:
    _validate_observation_length(observation)
    apf_config = config or APFConfig()
    values = tuple(float(value) for value in observation)
    attraction = attractive_force(_attraction_target(values, apf_config), apf_config)
    repulsion = _combined_repulsive_force(values, apf_config)
    combined = tuple(attraction[index] + repulsion[index] for index in range(3))
    return APFFeatureVector(
        attractive=attraction,
        repulsive=repulsion,
        combined=combined,
    )


def attractive_force(
    relative_goal: tuple[float, float, float],
    config: APFConfig | None = None,
) -> tuple[float, float, float]:
    apf_config = config or APFConfig()
    return _scale(_unit(relative_goal), apf_config.attractive_gain)


def repulsive_force(
    relative_obstacle: tuple[float, float, float],
    obstacle_radius: float,
    config: APFConfig | None = None,
) -> tuple[float, float, float]:
    apf_config = config or APFConfig()
    radius = float(obstacle_radius)
    if radius <= 0.0:
        return (0.0, 0.0, 0.0)

    distance = _norm(relative_obstacle)
    if distance <= apf_config.epsilon:
        direction = (-1.0, 0.0, 0.0)
    else:
        direction = _scale(_unit(relative_obstacle), -1.0)
    safety_distance = max(distance - radius, apf_config.epsilon)
    if safety_distance >= apf_config.influence_radius:
        return (0.0, 0.0, 0.0)

    magnitude = apf_config.repulsive_gain * (
        (1.0 / safety_distance) - (1.0 / apf_config.influence_radius)
    ) / (safety_distance * safety_distance)
    magnitude = min(magnitude, apf_config.max_repulsive_magnitude)
    return _scale(direction, magnitude)


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _unit(vector: tuple[float, ...]) -> tuple[float, float, float]:
    if len(vector) != 3:
        raise ValueError("vector must contain exactly 3 values")
    numeric = tuple(float(value) for value in vector)
    magnitude = _norm(numeric)
    if magnitude <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(value / magnitude for value in numeric)


def _scale(vector: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(value) * float(scalar) for value in vector)


def _combined_repulsive_force(observation: tuple[float, ...], config: APFConfig) -> tuple[float, float, float]:
    repulsion = (0.0, 0.0, 0.0)
    for relative_obstacle, obstacle_radius in _obstacle_slots(observation, config):
        force = repulsive_force(relative_obstacle, obstacle_radius, config)
        repulsion = tuple(repulsion[index] + force[index] for index in range(3))
    return repulsion


def _attraction_target(observation: tuple[float, ...], config: APFConfig) -> tuple[float, float, float]:
    relative_goal = observation[7:10]
    if config.visibility_planner_enabled:
        return visibility_waypoint_from_observation(
            observation,
            VisibilityPlannerConfig(
                clearance=config.visibility_clearance,
                samples=config.visibility_samples,
            ),
        )
    if not config.bypass_enabled:
        return relative_goal

    goal_xy_length = math.hypot(relative_goal[0], relative_goal[1])
    if goal_xy_length <= config.epsilon:
        return relative_goal

    forward = (relative_goal[0] / goal_xy_length, relative_goal[1] / goal_xy_length)
    lateral_axis = (-forward[1], forward[0])
    obstacles = []
    blockers = []
    for relative_obstacle, radius in _obstacle_slots(observation, config):
        along_path = relative_obstacle[0] * forward[0] + relative_obstacle[1] * forward[1]
        lateral_distance = relative_obstacle[0] * lateral_axis[0] + relative_obstacle[1] * lateral_axis[1]
        obstacle = (along_path, lateral_distance, radius)
        obstacles.append(obstacle)
        if along_path <= -radius or along_path >= goal_xy_length + radius:
            continue
        if abs(lateral_distance) <= radius + config.bypass_clearance:
            blockers.append(obstacle)

    if not blockers:
        return relative_goal

    primary = min(blockers, key=lambda item: (max(item[0], 0.0), abs(item[1])))
    side = _select_bypass_side(primary, obstacles, config)
    waypoint_forward = min(max(primary[0] + config.bypass_forward_margin, 0.0), goal_xy_length)
    target_x = forward[0] * waypoint_forward + lateral_axis[0] * side * config.bypass_lateral_offset
    target_y = forward[1] * waypoint_forward + lateral_axis[1] * side * config.bypass_lateral_offset
    return (target_x, target_y, relative_goal[2])


def _select_bypass_side(
    primary_blocker: tuple[float, float, float],
    obstacles: Sequence[tuple[float, float, float]],
    config: APFConfig,
) -> float:
    positive_clearance = _minimum_side_clearance(1.0, obstacles, config)
    negative_clearance = _minimum_side_clearance(-1.0, obstacles, config)
    if positive_clearance > negative_clearance + config.epsilon:
        return 1.0
    if negative_clearance > positive_clearance + config.epsilon:
        return -1.0

    primary_lateral = primary_blocker[1]
    if primary_lateral < -config.epsilon:
        return 1.0
    if primary_lateral > config.epsilon:
        return -1.0
    return 1.0


def _minimum_side_clearance(
    side: float,
    obstacles: Sequence[tuple[float, float, float]],
    config: APFConfig,
) -> float:
    target_lateral = float(side) * config.bypass_lateral_offset
    clearance = math.inf
    for along_path, lateral_distance, radius in obstacles:
        if along_path <= -radius:
            continue
        clearance = min(clearance, abs(target_lateral - lateral_distance) - radius)
    return clearance


def _obstacle_slots(
    observation: tuple[float, ...],
    config: APFConfig,
) -> tuple[tuple[tuple[float, float, float], float], ...]:
    if len(observation) == 15:
        return ((observation[10:13], observation[13]),)

    slots = []
    for offset in range(15, len(observation), 4):
        radius = observation[offset + 3]
        if radius <= 0.0:
            continue
        relative_obstacle = observation[offset : offset + 3]
        if config.ignore_obstacles_behind and relative_obstacle[0] < -radius:
            continue
        slots.append((relative_obstacle, radius))
    return tuple(slots)


def _validate_observation_length(observation: Sequence[float]) -> None:
    if len(observation) < 15 or (len(observation) - 15) % 4 != 0:
        raise ValueError("observation must contain 15 values plus zero or more 4-value obstacle slots")


def _set_positive_float(instance: object, name: str, value: float) -> None:
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be positive")
    object.__setattr__(instance, name, numeric_value)


def _set_non_negative_float(instance: object, name: str, value: float) -> None:
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    object.__setattr__(instance, name, numeric_value)
