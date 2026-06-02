from __future__ import annotations

from dataclasses import dataclass

Vector3 = tuple[float, float, float]


def _validate_vector3(name: str, value: tuple[float, ...]) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class DroneState:
    position: Vector3
    velocity: Vector3
    yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _validate_vector3("position", self.position))
        object.__setattr__(self, "velocity", _validate_vector3("velocity", self.velocity))
        object.__setattr__(self, "yaw", float(self.yaw))


@dataclass(frozen=True)
class DroneAction:
    speed: float
    heading_delta: float
    climb_rate: float


@dataclass(frozen=True)
class ObstacleState:
    position: Vector3
    radius: float
    velocity: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _validate_vector3("position", self.position))
        object.__setattr__(self, "velocity", _validate_vector3("velocity", self.velocity))
        if self.radius <= 0:
            raise ValueError("radius must be positive")


@dataclass(frozen=True)
class RewardBreakdown:
    arrive: float
    approach: float
    obstacle: float
    smoothness: float
    timeliness: float

    @property
    def total(self) -> float:
        return self.arrive + self.approach + self.obstacle + self.smoothness + self.timeliness


@dataclass(frozen=True)
class EpisodeMetrics:
    reached_goal: bool
    collided: bool
    timed_out: bool
    path_length: float
    path_smoothness: float
    minimum_safety_distance: float
    steps: int

    @property
    def success(self) -> bool:
        return self.reached_goal and not self.collided and not self.timed_out
