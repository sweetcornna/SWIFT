from __future__ import annotations

import math
from dataclasses import dataclass, field

from swift.rl.apf import APFConfig


@dataclass(frozen=True)
class HCAConfig:
    target_attention_heads: int = 4
    threat_attention_heads: int = 4
    embedding_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _set_positive_int(self, "target_attention_heads", self.target_attention_heads)
        _set_positive_int(self, "threat_attention_heads", self.threat_attention_heads)
        _set_positive_int(self, "embedding_dim", self.embedding_dim)
        _set_probability(self, "dropout", self.dropout)
        if self.embedding_dim % self.target_attention_heads != 0:
            raise ValueError("target_attention_heads must divide embedding_dim")
        if self.embedding_dim % self.threat_attention_heads != 0:
            raise ValueError("threat_attention_heads must divide embedding_dim")


@dataclass(frozen=True)
class HCAObservationAdapterConfig:
    observation_dim: int = 15
    self_dim: int = 7
    target_dim: int = 4
    threat_dim: int = 4

    def __post_init__(self) -> None:
        _set_exact_int(self, "observation_dim", self.observation_dim, 15)
        _set_exact_int(self, "self_dim", self.self_dim, 7)
        _set_exact_int(self, "target_dim", self.target_dim, 4)
        _set_exact_int(self, "threat_dim", self.threat_dim, 4)


@dataclass(frozen=True)
class HCAActorCriticConfig:
    observation: HCAObservationAdapterConfig = field(default_factory=HCAObservationAdapterConfig)
    action_dim: int = 3
    embedding_dim: int = 64
    target_attention_heads: int = 4
    threat_attention_heads: int = 4
    hidden_sizes: tuple[int, ...] = (64,)
    dropout: float = 0.0
    max_heading_delta: float = 0.5
    log_std_init: float = -0.5
    apf_config: APFConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, HCAObservationAdapterConfig):
            try:
                observation = HCAObservationAdapterConfig(
                    observation_dim=self.observation.observation_dim,
                    self_dim=self.observation.self_dim,
                    target_dim=self.observation.target_dim,
                    threat_dim=self.observation.threat_dim,
                )
            except AttributeError as exc:
                raise TypeError("observation must be an HCAObservationAdapterConfig") from exc
            object.__setattr__(self, "observation", observation)
        _set_exact_int(self, "action_dim", self.action_dim, 3)
        _set_positive_int(self, "embedding_dim", self.embedding_dim)
        _set_positive_int(self, "target_attention_heads", self.target_attention_heads)
        _set_positive_int(self, "threat_attention_heads", self.threat_attention_heads)
        if self.embedding_dim % self.target_attention_heads != 0:
            raise ValueError("target_attention_heads must divide embedding_dim")
        if self.embedding_dim % self.threat_attention_heads != 0:
            raise ValueError("threat_attention_heads must divide embedding_dim")
        object.__setattr__(self, "hidden_sizes", tuple(int(size) for size in self.hidden_sizes))
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        for size in self.hidden_sizes:
            if size <= 0:
                raise ValueError("hidden_sizes values must be positive")
        _set_probability(self, "dropout", self.dropout)
        _set_positive_float(self, "max_heading_delta", self.max_heading_delta)
        _set_finite_float(self, "log_std_init", self.log_std_init)
        if self.apf_config is not None and not isinstance(self.apf_config, APFConfig):
            try:
                apf_config = APFConfig(
                    attractive_gain=self.apf_config.attractive_gain,
                    repulsive_gain=self.apf_config.repulsive_gain,
                    influence_radius=self.apf_config.influence_radius,
                    max_repulsive_magnitude=self.apf_config.max_repulsive_magnitude,
                    epsilon=self.apf_config.epsilon,
                    ignore_obstacles_behind=getattr(self.apf_config, "ignore_obstacles_behind", False),
                    bypass_enabled=getattr(self.apf_config, "bypass_enabled", False),
                    bypass_lateral_offset=getattr(self.apf_config, "bypass_lateral_offset", 0.25),
                    bypass_forward_margin=getattr(self.apf_config, "bypass_forward_margin", 0.08),
                    bypass_clearance=getattr(self.apf_config, "bypass_clearance", 0.16),
                    visibility_planner_enabled=getattr(
                        self.apf_config,
                        "visibility_planner_enabled",
                        False,
                    ),
                    visibility_clearance=getattr(
                        self.apf_config,
                        "visibility_clearance",
                        0.18,
                    ),
                    visibility_samples=getattr(self.apf_config, "visibility_samples", 16),
                    policy_residual_scale=getattr(
                        self.apf_config,
                        "policy_residual_scale",
                        1.0,
                    ),
                )
            except AttributeError as exc:
                raise TypeError("apf_config must be an APFConfig") from exc
            object.__setattr__(self, "apf_config", apf_config)


def _set_exact_int(instance: object, name: str, value: int, expected: int) -> None:
    integer_value = int(value)
    if integer_value != expected:
        raise ValueError(f"{name} must be {expected}")
    object.__setattr__(instance, name, integer_value)


def _set_positive_int(instance: object, name: str, value: int) -> None:
    integer_value = int(value)
    if integer_value <= 0:
        raise ValueError(f"{name} must be positive")
    object.__setattr__(instance, name, integer_value)


def _set_finite_float(instance: object, name: str, value: float) -> None:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    object.__setattr__(instance, name, numeric_value)


def _set_positive_float(instance: object, name: str, value: float) -> None:
    _set_finite_float(instance, name, value)
    if getattr(instance, name) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _set_probability(instance: object, name: str, value: float) -> None:
    _set_finite_float(instance, name, value)
    numeric_value = getattr(instance, name)
    if numeric_value < 0.0 or numeric_value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
