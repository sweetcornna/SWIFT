from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from typing import Any

from swift.core import ObstacleState
from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings
from swift.rl.apf import APFConfig, apf_features_from_observation
from swift.rl.mlp_baseline import MLPBaselinePolicy, MLPBaselinePolicyConfig


@dataclass(frozen=True)
class Stage1Scenario:
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal: tuple[float, float, float] = (10.0, 0.0, 0.0)
    obstacles: tuple[ObstacleState, ...] = (
        ObstacleState(position=(4.0, 0.0, 0.0), radius=0.6),
    )
    max_steps: int = 60
    goal_radius: float = 0.5
    safety_margin: float = 0.25

    def build_settings(self) -> SimpleAvoidanceSettings:
        return SimpleAvoidanceSettings(
            start=self.start,
            goal=self.goal,
            obstacles=self.obstacles,
            max_steps=self.max_steps,
            goal_radius=self.goal_radius,
            safety_margin=self.safety_margin,
        )


@dataclass(frozen=True)
class PolicySearchSpace:
    max_speed: tuple[float, ...] = (1.0, 0.8)
    max_heading_delta: tuple[float, ...] = (0.5, 0.6)
    obstacle_avoidance_distance: tuple[float, ...] = (2.0, 4.0)
    avoidance_heading_delta: tuple[float, ...] = (0.4,)
    max_climb_rate: tuple[float, ...] = (0.5,)

    def __post_init__(self) -> None:
        _set_float_tuple(self, "max_speed", self.max_speed)
        _set_float_tuple(self, "max_heading_delta", self.max_heading_delta)
        _set_float_tuple(self, "obstacle_avoidance_distance", self.obstacle_avoidance_distance)
        _set_float_tuple(self, "avoidance_heading_delta", self.avoidance_heading_delta)
        _set_float_tuple(self, "max_climb_rate", self.max_climb_rate)


@dataclass(frozen=True)
class TuningRunConfig:
    scenario: Stage1Scenario = field(default_factory=Stage1Scenario)
    search_space: PolicySearchSpace = field(default_factory=PolicySearchSpace)
    minimum_safety_distance: float = 0.30
    candidate_limit: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_safety_distance", float(self.minimum_safety_distance))
        if not math.isfinite(self.minimum_safety_distance) or self.minimum_safety_distance < 0.0:
            raise ValueError("minimum_safety_distance must be finite and non-negative")
        if self.candidate_limit is not None and self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive or None")


def evaluate_policy(
    settings: SimpleAvoidanceSettings,
    policy_config: MLPBaselinePolicyConfig,
) -> dict[str, object]:
    config = _coerce_policy_config(policy_config)
    env_settings = settings.replace(max_speed=config.max_speed, max_climb_rate=config.max_climb_rate)
    env = SimpleAvoidanceEnv(env_settings)
    policy = MLPBaselinePolicy(config)
    observation, _ = env.reset()
    info: dict[str, Any] = {}
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)

    metrics = info["episode_metrics"]
    apf_features = _apf_feature_dict(observation)
    return {
        "success": bool(metrics.success),
        "collided": bool(metrics.collided),
        "timed_out": bool(metrics.timed_out),
        "steps": int(metrics.steps),
        "path_length": float(metrics.path_length),
        "path_smoothness": float(metrics.path_smoothness),
        "minimum_safety_distance": float(metrics.minimum_safety_distance),
        "apf": apf_features,
    }


def run_stage1_policy_search(config: TuningRunConfig | None = None) -> dict[str, object]:
    run_config = config or TuningRunConfig()
    settings = run_config.scenario.build_settings()
    baseline_config = MLPBaselinePolicyConfig()
    baseline_metrics = evaluate_policy(settings, baseline_config)
    candidates = [
        _candidate_result(settings, policy_config, run_config.minimum_safety_distance)
        for policy_config in _policy_configs(run_config.search_space)
    ]
    ranked_candidates = sorted(candidates, key=_candidate_sort_key)
    accepted_candidates = [candidate for candidate in ranked_candidates if candidate["accepted"]]
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = rank

    candidate_limit = run_config.candidate_limit
    output_candidates = ranked_candidates if candidate_limit is None else ranked_candidates[:candidate_limit]
    return {
        "scenario": _scenario_dict(run_config.scenario),
        "baseline": {
            "config": _policy_config_dict(baseline_config),
            **baseline_metrics,
        },
        "best_candidate": accepted_candidates[0] if accepted_candidates else None,
        "candidates": output_candidates,
        "acceptance": {
            "accepted": bool(accepted_candidates),
            "minimum_safety_distance": run_config.minimum_safety_distance,
            "candidate_count": len(ranked_candidates),
            "accepted_candidate_count": len(accepted_candidates),
        },
    }


def _candidate_result(
    settings: SimpleAvoidanceSettings,
    policy_config: MLPBaselinePolicyConfig,
    minimum_safety_distance: float,
) -> dict[str, object]:
    evaluation = evaluate_policy(settings, policy_config)
    apf = evaluation.pop("apf")
    metrics = evaluation
    accepted = (
        bool(metrics["success"])
        and not bool(metrics["collided"])
        and not bool(metrics["timed_out"])
        and float(metrics["minimum_safety_distance"]) >= minimum_safety_distance
    )
    return {
        "config": _policy_config_dict(policy_config),
        "metrics": metrics,
        "apf": apf,
        "accepted": accepted,
    }


def _candidate_sort_key(candidate: dict[str, object]) -> tuple[object, ...]:
    metrics = candidate["metrics"]
    config = candidate["config"]
    assert isinstance(metrics, dict)
    assert isinstance(config, dict)
    return (
        0 if candidate["accepted"] else 1,
        float(metrics["path_length"]),
        float(metrics["path_smoothness"]),
        -float(metrics["minimum_safety_distance"]),
        int(metrics["steps"]),
        float(config["max_speed"]),
        float(config["max_heading_delta"]),
        float(config["obstacle_avoidance_distance"]),
        float(config["avoidance_heading_delta"]),
        float(config["max_climb_rate"]),
    )


def _policy_configs(search_space: PolicySearchSpace) -> tuple[MLPBaselinePolicyConfig, ...]:
    return tuple(
        MLPBaselinePolicyConfig(
            max_speed=max_speed,
            max_heading_delta=max_heading_delta,
            max_climb_rate=max_climb_rate,
            obstacle_avoidance_distance=obstacle_avoidance_distance,
            avoidance_heading_delta=avoidance_heading_delta,
        )
        for max_speed, max_heading_delta, obstacle_avoidance_distance, avoidance_heading_delta, max_climb_rate in product(
            search_space.max_speed,
            search_space.max_heading_delta,
            search_space.obstacle_avoidance_distance,
            search_space.avoidance_heading_delta,
            search_space.max_climb_rate,
        )
    )


def _coerce_policy_config(policy_config: MLPBaselinePolicyConfig) -> MLPBaselinePolicyConfig:
    if isinstance(policy_config, MLPBaselinePolicyConfig):
        return policy_config
    return MLPBaselinePolicyConfig(**policy_config)


def _policy_config_dict(config: MLPBaselinePolicyConfig) -> dict[str, float]:
    return {
        "max_speed": float(config.max_speed),
        "max_heading_delta": float(config.max_heading_delta),
        "max_climb_rate": float(config.max_climb_rate),
        "obstacle_avoidance_distance": float(config.obstacle_avoidance_distance),
        "avoidance_heading_delta": float(config.avoidance_heading_delta),
    }


def _scenario_dict(scenario: Stage1Scenario) -> dict[str, object]:
    return {
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "obstacles": [_obstacle_dict(obstacle) for obstacle in scenario.obstacles],
        "max_steps": int(scenario.max_steps),
        "goal_radius": float(scenario.goal_radius),
        "safety_margin": float(scenario.safety_margin),
    }


def _obstacle_dict(obstacle: ObstacleState) -> dict[str, object]:
    return {
        "position": list(obstacle.position),
        "radius": float(obstacle.radius),
        "velocity": list(obstacle.velocity),
    }


def _apf_feature_dict(observation: tuple[float, ...]) -> dict[str, object]:
    features = apf_features_from_observation(observation, APFConfig())
    return {
        "attractive": list(features.attractive),
        "repulsive": list(features.repulsive),
        "combined": list(features.combined),
        "combined_norm": math.sqrt(sum(value * value for value in features.combined)),
        "as_tuple": list(features.as_tuple()),
    }


def _set_float_tuple(instance: object, name: str, values: tuple[float, ...]) -> None:
    numeric_values = tuple(float(value) for value in values)
    if not numeric_values:
        raise ValueError(f"{name} must contain at least one value")
    if any(not math.isfinite(value) or value < 0.0 for value in numeric_values):
        raise ValueError(f"{name} values must be finite and non-negative")
    object.__setattr__(instance, name, numeric_values)
