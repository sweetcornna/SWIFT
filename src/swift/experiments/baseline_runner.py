from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from swift.core import EpisodeMetrics, compute_path_length, compute_path_smoothness
from swift.rl import MLPBaselinePolicy, MLPBaselinePolicyConfig

try:
    from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings
except ImportError:
    SimpleAvoidanceEnv = None  # type: ignore[assignment]
    SimpleAvoidanceSettings = None  # type: ignore[assignment]

VARIANT_NAME = "ppo_mlp_contract_baseline"


@dataclass(frozen=True)
class BaselineRunConfig:
    episodes: int = 10
    env_settings: object = field(default_factory=lambda: _default_env_settings())
    policy_config: MLPBaselinePolicyConfig = field(default_factory=MLPBaselinePolicyConfig)


def run_baseline_episodes(config: BaselineRunConfig | None = None) -> dict[str, object]:
    run_config = config or BaselineRunConfig()
    if run_config.episodes <= 0:
        raise ValueError("episodes must be positive")

    env = _make_env(run_config.env_settings)
    policy = MLPBaselinePolicy(run_config.policy_config)
    episode_metrics = [_run_episode(env, policy) for _ in range(run_config.episodes)]

    episodes_detail = [_episode_detail(metrics) for metrics in episode_metrics]
    return {
        "variant": VARIANT_NAME,
        "episodes": run_config.episodes,
        "success_rate": _rate(metrics.success for metrics in episode_metrics),
        "collision_rate": _rate(metrics.collided for metrics in episode_metrics),
        "timeout_rate": _rate(metrics.timed_out for metrics in episode_metrics),
        "average_path_length": _average(metrics.path_length for metrics in episode_metrics),
        "average_path_smoothness": _average(metrics.path_smoothness for metrics in episode_metrics),
        "minimum_safety_distance": min(metrics.minimum_safety_distance for metrics in episode_metrics),
        "episodes_detail": episodes_detail,
    }


def _run_episode(env: object, policy: MLPBaselinePolicy) -> EpisodeMetrics:
    observation, _ = env.reset()
    observation = _as_observation_tuple(observation)
    path = [_position_from_observation(observation)]
    safety_distances = [_safety_distance_from_observation(observation)]
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    steps = 0

    while not terminated and not truncated:
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)
        observation = _as_observation_tuple(observation)
        path.append(_position_from_observation(observation))
        safety_distances.append(_safety_distance_from_observation(observation))
        steps += 1

    reached_goal = bool(info.get("reached_goal", info.get("success", False)))
    collided = bool(info.get("collided", info.get("collision", False)))
    timed_out = bool(info.get("timed_out", truncated and not reached_goal and not collided))

    return EpisodeMetrics(
        reached_goal=reached_goal,
        collided=collided,
        timed_out=timed_out,
        path_length=compute_path_length(path),
        path_smoothness=compute_path_smoothness(path),
        minimum_safety_distance=min(safety_distances),
        steps=steps,
    )


def _default_env_settings() -> object:
    settings_cls = SimpleAvoidanceSettings
    if settings_cls is None:
        from swift.envs import SimpleAvoidanceSettings as settings_cls
    return settings_cls()


def _make_env(env_settings: object) -> object:
    env_cls = SimpleAvoidanceEnv
    if env_cls is None:
        from swift.envs import SimpleAvoidanceEnv as env_cls
    return env_cls(env_settings)


def _as_observation_tuple(observation: tuple[float, ...]) -> tuple[float, ...]:
    if len(observation) != 15:
        raise ValueError("environment observation must contain exactly 15 values")
    return tuple(float(value) for value in observation)


def _position_from_observation(observation: tuple[float, ...]) -> tuple[float, float, float]:
    return observation[0], observation[1], observation[2]


def _safety_distance_from_observation(observation: tuple[float, ...]) -> float:
    obstacle_relative = observation[10:13]
    obstacle_radius = observation[13]
    obstacle_distance = math.sqrt(sum(value * value for value in obstacle_relative))
    return obstacle_distance - obstacle_radius


def _episode_detail(metrics: EpisodeMetrics) -> dict[str, object]:
    detail = asdict(metrics)
    detail["success"] = metrics.success
    return detail


def _rate(values: object) -> float:
    counted_values = tuple(bool(value) for value in values)
    return sum(counted_values) / len(counted_values)


def _average(values: object) -> float:
    counted_values = tuple(float(value) for value in values)
    return sum(counted_values) / len(counted_values)
