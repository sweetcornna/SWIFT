from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExperimentMetric(StrEnum):
    SUCCESS_RATE = "success_rate"
    COLLISION_RATE = "collision_rate"
    AVERAGE_PATH_LENGTH = "average_path_length"
    PATH_SMOOTHNESS = "path_smoothness"
    CONVERGENCE_EPISODES = "convergence_episodes"
    MINIMUM_INTER_DRONE_DISTANCE = "minimum_inter_drone_distance"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    variants: tuple[str, ...]
    metrics: tuple[ExperimentMetric, ...]
