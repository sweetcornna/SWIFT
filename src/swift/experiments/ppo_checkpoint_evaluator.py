from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings


@dataclass(frozen=True)
class PPOCheckpointEvaluationConfig:
    checkpoint_path: Path
    output: Path | None = None
    episodes: int = 3
    seed: int = 0
    environment: SimpleAvoidanceSettings = field(default_factory=SimpleAvoidanceSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def run_ppo_checkpoint_evaluation(config: PPOCheckpointEvaluationConfig) -> dict[str, Any]:
    model, checkpoint, deterministic_action, policy_family = _load_checkpoint_policy(config.checkpoint_path)
    episodes = [
        _evaluate_episode(
            model=model,
            deterministic_action=deterministic_action,
            settings=config.environment,
            seed=config.seed + index,
        )
        for index in range(config.episodes)
    ]
    summary = {
        "schema_version": 1,
        "record_type": "ppo_checkpoint_evaluation",
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_record_type": str(checkpoint.get("record_type", "")),
        "policy_family": policy_family,
        "checkpoint_training": dict(checkpoint.get("result", {})),
        "episodes_requested": config.episodes,
        "seed": config.seed,
        "metrics": _aggregate_metrics(episodes),
        "episodes": episodes,
        "artifacts": {
            "summary_json": str(config.output) if config.output is not None else "",
        },
    }
    _assert_json_safe(summary)
    if config.output is not None:
        serialized = json.dumps(summary, allow_nan=False, sort_keys=True)
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(f"{serialized}\n", encoding="utf-8")
        return json.loads(config.output.read_text(encoding="utf-8"))
    return json.loads(json.dumps(summary, allow_nan=False, sort_keys=True))


def _load_checkpoint_policy(path: Path) -> tuple[Any, dict[str, Any], Any, str]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    record_type = checkpoint.get("record_type")
    if record_type == "ppo_checkpoint":
        from swift.rl.torch_ppo import deterministic_action, load_ppo_mlp_checkpoint

        model, loaded_checkpoint = load_ppo_mlp_checkpoint(path)
        return model, loaded_checkpoint, deterministic_action, "ppo_mlp"
    if record_type == "ppo_hca_checkpoint":
        from swift.rl.torch_hca_ppo import deterministic_hca_action, load_ppo_hca_checkpoint

        model, loaded_checkpoint = load_ppo_hca_checkpoint(path)
        policy_family = "ppo_hca_apf" if isinstance(loaded_checkpoint.get("network_config", {}).get("apf_config"), dict) else "ppo_hca"
        return model, loaded_checkpoint, deterministic_hca_action, policy_family
    raise ValueError("checkpoint record_type must be ppo_checkpoint or ppo_hca_checkpoint")


def _evaluate_episode(
    *,
    model: Any,
    deterministic_action: Any,
    settings: SimpleAvoidanceSettings,
    seed: int,
) -> dict[str, Any]:
    env = SimpleAvoidanceEnv(settings)
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    info: dict[str, Any] = {}
    while not terminated and not truncated:
        action = deterministic_action(model, observation, env.settings, model.config)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    metrics = info["episode_metrics"]
    return {
        "seed": seed,
        "success": bool(metrics.success),
        "collided": bool(metrics.collided),
        "timed_out": bool(metrics.timed_out),
        "steps": int(metrics.steps),
        "return": total_reward,
        "path_length": float(metrics.path_length),
        "path_smoothness": float(metrics.path_smoothness),
        "minimum_safety_distance": float(metrics.minimum_safety_distance),
    }


def _aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    total = len(episodes)
    return {
        "success_rate": _rate(sum(1 for episode in episodes if episode["success"]), total),
        "collision_rate": _rate(sum(1 for episode in episodes if episode["collided"]), total),
        "timeout_rate": _rate(sum(1 for episode in episodes if episode["timed_out"]), total),
        "average_episode_return": _mean(float(episode["return"]) for episode in episodes),
        "average_steps": _mean(float(episode["steps"]) for episode in episodes),
        "average_minimum_safety_distance": _mean(
            float(episode["minimum_safety_distance"]) for episode in episodes
        ),
    }


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(count) / float(total)


def _mean(values: Any) -> float:
    numeric_values = tuple(float(value) for value in values)
    if not numeric_values:
        return 0.0
    return float(sum(numeric_values)) / float(len(numeric_values))


def _assert_json_safe(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evaluation contains a non-finite float")
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe(item)
