from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from swift.rl.hca import HCAActorCriticConfig


class TorchUnavailableError(RuntimeError):
    """Raised when the optional Torch PPO backend cannot be imported."""


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 2048
    minibatch_size: int = 64
    update_epochs: int = 10
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        _set_positive_int(self, "rollout_steps", self.rollout_steps)
        _set_positive_int(self, "minibatch_size", self.minibatch_size)
        _set_positive_int(self, "update_epochs", self.update_epochs)
        _set_positive_float(self, "clip_range", self.clip_range)
        _set_probability(self, "gamma", self.gamma)
        _set_probability(self, "gae_lambda", self.gae_lambda)
        _set_non_negative_float(self, "entropy_coef", self.entropy_coef)
        _set_non_negative_float(self, "value_loss_coef", self.value_loss_coef)
        _set_positive_float(self, "max_grad_norm", self.max_grad_norm)
        if self.minibatch_size > self.rollout_steps:
            raise ValueError("minibatch_size must be <= rollout_steps")


@dataclass(frozen=True)
class MLPActorCriticConfig:
    observation_dim: int = 15
    action_dim: int = 3
    hidden_sizes: tuple[int, ...] = (64, 64)
    max_heading_delta: float = 0.5
    log_std_init: float = -0.5

    def __post_init__(self) -> None:
        _set_exact_int(self, "observation_dim", self.observation_dim, 15)
        _set_exact_int(self, "action_dim", self.action_dim, 3)
        object.__setattr__(self, "hidden_sizes", tuple(int(size) for size in self.hidden_sizes))
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        for size in self.hidden_sizes:
            if size <= 0:
                raise ValueError("hidden_sizes values must be positive")
        _set_positive_float(self, "max_heading_delta", self.max_heading_delta)
        _set_finite_float(self, "log_std_init", self.log_std_init)


@dataclass(frozen=True)
class PPOTrainingConfig(PPOConfig):
    total_timesteps: int = 128
    rollout_steps: int = 32
    minibatch_size: int = 16
    update_epochs: int = 1
    learning_rate: float = 3e-4
    seed: int = 0
    torch_num_threads: int = 1
    network: MLPActorCriticConfig = field(default_factory=MLPActorCriticConfig)
    checkpoint_path: Path | None = None
    history_path: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _set_positive_int(self, "total_timesteps", self.total_timesteps)
        _set_positive_float(self, "learning_rate", self.learning_rate)
        _set_non_negative_int(self, "seed", self.seed)
        _set_positive_int(self, "torch_num_threads", self.torch_num_threads)
        if self.total_timesteps < self.rollout_steps:
            raise ValueError("total_timesteps must be >= rollout_steps")
        if not isinstance(self.network, MLPActorCriticConfig):
            try:
                network = MLPActorCriticConfig(
                    observation_dim=self.network.observation_dim,
                    action_dim=self.network.action_dim,
                    hidden_sizes=self.network.hidden_sizes,
                    max_heading_delta=self.network.max_heading_delta,
                    log_std_init=self.network.log_std_init,
                )
            except AttributeError as exc:
                raise TypeError("network must be an MLPActorCriticConfig") from exc
            object.__setattr__(self, "network", network)
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.history_path is not None:
            object.__setattr__(self, "history_path", Path(self.history_path))


@dataclass(frozen=True)
class HCAPPOTrainingConfig(PPOConfig):
    total_timesteps: int = 128
    rollout_steps: int = 32
    minibatch_size: int = 16
    update_epochs: int = 1
    learning_rate: float = 3e-4
    seed: int = 0
    torch_num_threads: int = 1
    network: HCAActorCriticConfig = field(default_factory=HCAActorCriticConfig)
    checkpoint_path: Path | None = None
    history_path: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _set_positive_int(self, "total_timesteps", self.total_timesteps)
        _set_positive_float(self, "learning_rate", self.learning_rate)
        _set_non_negative_int(self, "seed", self.seed)
        _set_positive_int(self, "torch_num_threads", self.torch_num_threads)
        if self.total_timesteps < self.rollout_steps:
            raise ValueError("total_timesteps must be >= rollout_steps")
        if not isinstance(self.network, HCAActorCriticConfig):
            try:
                network = HCAActorCriticConfig(
                    observation=self.network.observation,
                    action_dim=self.network.action_dim,
                    embedding_dim=self.network.embedding_dim,
                    target_attention_heads=self.network.target_attention_heads,
                    threat_attention_heads=self.network.threat_attention_heads,
                    hidden_sizes=self.network.hidden_sizes,
                    dropout=self.network.dropout,
                    max_heading_delta=self.network.max_heading_delta,
                    log_std_init=self.network.log_std_init,
                    apf_config=self.network.apf_config,
                )
            except AttributeError as exc:
                raise TypeError("network must be an HCAActorCriticConfig") from exc
            object.__setattr__(self, "network", network)
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.history_path is not None:
            object.__setattr__(self, "history_path", Path(self.history_path))


@dataclass(frozen=True)
class PPOTrainingResult:
    total_timesteps: int
    updates: int
    episodes_completed: int
    average_episode_return: float
    success_rate: float
    collision_rate: float
    timeout_rate: float
    final_policy_loss: float
    final_value_loss: float
    final_entropy: float
    checkpoint_path: str | None = None
    history_path: str | None = None


def train_ppo_mlp(
    make_env: Callable[[], Any],
    config: PPOTrainingConfig | None = None,
) -> PPOTrainingResult:
    training_config = config or PPOTrainingConfig()
    try:
        backend = import_module(".torch_ppo", __package__)
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or (exc.name is not None and exc.name.startswith("torch.")):
            raise TorchUnavailableError(
                "torch is required for train_ppo_mlp; install the train extra"
            ) from exc
        raise
    return backend.train_ppo_mlp(make_env, training_config)


def train_ppo_hca(
    make_env: Callable[[], Any],
    config: HCAPPOTrainingConfig | None = None,
) -> PPOTrainingResult:
    training_config = config or HCAPPOTrainingConfig()
    try:
        backend = import_module(".torch_hca_ppo", __package__)
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or (exc.name is not None and exc.name.startswith("torch.")):
            raise TorchUnavailableError(
                "torch is required for train_ppo_hca; install the train extra"
            ) from exc
        raise
    return backend.train_ppo_hca(make_env, training_config)


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


def _set_non_negative_int(instance: object, name: str, value: int) -> None:
    integer_value = int(value)
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
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


def _set_non_negative_float(instance: object, name: str, value: float) -> None:
    _set_finite_float(instance, name, value)
    if getattr(instance, name) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _set_probability(instance: object, name: str, value: float) -> None:
    _set_finite_float(instance, name, value)
    numeric_value = getattr(instance, name)
    if numeric_value < 0.0 or numeric_value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
