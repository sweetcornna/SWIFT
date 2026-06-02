from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APFConfig:
    attractive_gain: float = 1.0
    repulsive_gain: float = 1.0
    influence_radius: float = 2.0
    epsilon: float = 1e-6
