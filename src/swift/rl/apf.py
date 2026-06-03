from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class APFConfig:
    attractive_gain: float = 1.0
    repulsive_gain: float = 1.0
    influence_radius: float = 2.0
    max_repulsive_magnitude: float = 10.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        _set_non_negative_float(self, "attractive_gain", self.attractive_gain)
        _set_non_negative_float(self, "repulsive_gain", self.repulsive_gain)
        _set_positive_float(self, "influence_radius", self.influence_radius)
        _set_positive_float(self, "max_repulsive_magnitude", self.max_repulsive_magnitude)
        _set_positive_float(self, "epsilon", self.epsilon)


@dataclass(frozen=True)
class APFFeatureVector:
    attractive: tuple[float, float, float]
    repulsive: tuple[float, float, float]
    combined: tuple[float, float, float]

    def as_tuple(self) -> tuple[float, ...]:
        return (*self.attractive, *self.repulsive, *self.combined)


def apf_features_from_observation(
    observation: tuple[float, ...],
    config: APFConfig | None = None,
) -> APFFeatureVector:
    if len(observation) != 15:
        raise ValueError("observation must contain exactly 15 values")
    apf_config = config or APFConfig()
    values = tuple(float(value) for value in observation)
    attraction = attractive_force(values[7:10], apf_config)
    repulsion = repulsive_force(values[10:13], values[13], apf_config)
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
