from __future__ import annotations

import math

from swift.core.types import Vector3


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def compute_path_length(points: list[Vector3] | tuple[Vector3, ...]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_distance(start, end) for start, end in zip(points, points[1:]))


def compute_path_smoothness(points: list[Vector3] | tuple[Vector3, ...]) -> float:
    if len(points) < 3:
        return 0.0

    total_turn = 0.0
    for previous, current, following in zip(points, points[1:], points[2:]):
        v1 = tuple(current[i] - previous[i] for i in range(3))
        v2 = tuple(following[i] - current[i] for i in range(3))
        n1 = math.sqrt(sum(value * value for value in v1))
        n2 = math.sqrt(sum(value * value for value in v2))
        if n1 == 0.0 or n2 == 0.0:
            continue
        cos_angle = sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        total_turn += math.acos(cos_angle)
    return total_turn


def compute_minimum_distance(
    ownship: list[Vector3] | tuple[Vector3, ...],
    intruder: list[Vector3] | tuple[Vector3, ...],
) -> float:
    if not ownship or not intruder:
        raise ValueError("ownship and intruder tracks must not be empty")
    return min(_distance(a, b) for a in ownship for b in intruder)
