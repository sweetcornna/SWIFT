from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from swift.core import DroneAction
from swift.envs import SimpleAvoidanceSettings
from swift.rl.ppo import MLPActorCriticConfig, PPOPhaseMetrics, PPOTrainingConfig, PPOTrainingResult


class MLPActorCritic(nn.Module):
    def __init__(self, config: MLPActorCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPActorCriticConfig()
        layers: list[nn.Module] = []
        input_dim = self.config.observation_dim
        for hidden_size in self.config.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(input_dim, self.config.action_dim)
        self.value_head = nn.Linear(input_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((self.config.action_dim,), float(self.config.log_std_init), dtype=torch.float32)
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        action_means = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)
        return action_means, values


def sample_action(
    model: MLPActorCritic,
    observation: Sequence[float],
    settings: SimpleAvoidanceSettings,
    network_config: MLPActorCriticConfig,
) -> tuple[DroneAction, torch.Tensor, torch.Tensor, torch.Tensor]:
    observation_tensor = _observation_tensor(observation)
    model.eval()
    with torch.no_grad():
        action_means, values = model(observation_tensor.unsqueeze(0))
        distribution = _action_distribution(model, action_means)
        raw_action = distribution.sample()[0]
        logprob = distribution.log_prob(raw_action.unsqueeze(0)).sum(dim=-1)[0]
        value = values[0]

    return (
        _raw_action_to_drone_action(raw_action, settings, network_config),
        raw_action.detach().cpu(),
        logprob.detach().cpu(),
        value.detach().cpu(),
    )


def deterministic_action(
    model: MLPActorCritic,
    observation: Sequence[float],
    settings: SimpleAvoidanceSettings,
    network_config: MLPActorCriticConfig,
) -> DroneAction:
    observation_tensor = _observation_tensor(observation)
    model.eval()
    with torch.no_grad():
        action_means, _ = model(observation_tensor.unsqueeze(0))
    return _raw_action_to_drone_action(action_means[0].detach().cpu(), settings, network_config)


def load_ppo_mlp_checkpoint(path: str | Path) -> tuple[MLPActorCritic, dict[str, Any]]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("record_type") != "ppo_checkpoint":
        raise ValueError("checkpoint record_type must be ppo_checkpoint")
    network_config = MLPActorCriticConfig(**checkpoint["network_config"])
    model = MLPActorCritic(network_config).to(torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = rewards.to(dtype=torch.float32)
    values = values.to(dtype=torch.float32)
    dones = dones.to(dtype=torch.bool)
    next_value = next_value.to(dtype=torch.float32).reshape(())
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), dtype=torch.float32, device=rewards.device)

    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = (~dones[step]).to(dtype=torch.float32)
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage

    returns = advantages + values
    return returns, advantages


def train_ppo_mlp(
    make_env: Callable[[], Any],
    config: PPOTrainingConfig,
) -> PPOTrainingResult:
    torch.set_num_threads(config.torch_num_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    model = MLPActorCritic(config.network).to(torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    env = make_env()
    try:
        return _train_ppo_mlp_open_env(env=env, model=model, optimizer=optimizer, config=config)
    finally:
        _close_env(env)


def _train_ppo_mlp_open_env(
    *,
    env: Any,
    model: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOTrainingConfig,
) -> PPOTrainingResult:
    _set_training_progress(env, 0, config.total_timesteps)
    observation, _ = env.reset(seed=config.seed)

    total_timesteps = 0
    updates = 0
    episodes_completed = 0
    episode_returns: list[float] = []
    current_episode_return = 0.0
    successes = 0
    collisions = 0
    timeouts = 0
    phase_episodes: list[dict[str, Any]] = []
    final_policy_loss = 0.0
    final_value_loss = 0.0
    final_entropy = 0.0
    if config.history_path is not None:
        _prepare_jsonl(config.history_path)

    while total_timesteps < config.total_timesteps:
        rollout_length = min(config.rollout_steps, config.total_timesteps - total_timesteps)
        rollout = _collect_rollout(
            env=env,
            model=model,
            observation=observation,
            config=config,
            rollout_length=rollout_length,
            seed_offset=episodes_completed,
            starting_timestep=total_timesteps,
            current_episode_return=current_episode_return,
        )
        observation = rollout["observation"]
        total_timesteps += rollout_length
        episodes_completed += rollout["episodes_completed"]
        episode_returns.extend(rollout["episode_returns"])
        current_episode_return = rollout["current_episode_return"]
        successes += rollout["successes"]
        collisions += rollout["collisions"]
        timeouts += rollout["timeouts"]
        phase_episodes.extend(rollout["phase_episodes"])

        with torch.no_grad():
            _, next_values = model(_observation_tensor(observation).unsqueeze(0))
            next_value = next_values[0]

        rewards = torch.tensor(rollout["rewards"], dtype=torch.float32)
        values = torch.stack(rollout["values"]).to(dtype=torch.float32)
        dones = torch.tensor(rollout["dones"], dtype=torch.bool)
        returns, advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = _normalize_advantages(advantages)

        observations = torch.stack(rollout["observations"]).to(dtype=torch.float32)
        actions = torch.stack(rollout["actions"]).to(dtype=torch.float32)
        old_logprobs = torch.stack(rollout["logprobs"]).to(dtype=torch.float32)
        batch_size = observations.shape[0]
        minibatch_size = min(config.minibatch_size, batch_size)

        model.train()
        for _ in range(config.update_epochs):
            permutation = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                indexes = permutation[start : start + minibatch_size]
                logprobs, entropy, predicted_values = _evaluate_actions(
                    model,
                    observations[indexes],
                    actions[indexes],
                )
                ratio = torch.exp(logprobs - old_logprobs[indexes])
                unclipped = ratio * advantages[indexes]
                clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages[
                    indexes
                ]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(predicted_values, returns[indexes])
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + config.value_loss_coef * value_loss
                    - config.entropy_coef * entropy_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                final_policy_loss = float(policy_loss.detach().item())
                final_value_loss = float(value_loss.detach().item())
                final_entropy = float(entropy_loss.detach().item())
        updates += 1
        phase_metrics = _summarize_phase_metrics(phase_episodes)
        if config.history_path is not None:
            _append_history(
                config.history_path,
                {
                    "schema_version": 1,
                    "record_type": "ppo_update",
                    "update": updates,
                    "total_timesteps": total_timesteps,
                    "episodes_completed": episodes_completed,
                    "success_rate": _rate(successes, episodes_completed),
                    "collision_rate": _rate(collisions, episodes_completed),
                    "timeout_rate": _rate(timeouts, episodes_completed),
                    "average_episode_return": _rate_sum(episode_returns),
                    "phase_metrics": [asdict(item) for item in phase_metrics],
                    "policy_loss": final_policy_loss,
                    "value_loss": final_value_loss,
                    "entropy": final_entropy,
                },
            )

    result = PPOTrainingResult(
        total_timesteps=total_timesteps,
        updates=updates,
        episodes_completed=episodes_completed,
        average_episode_return=_rate_sum(episode_returns),
        success_rate=_rate(successes, episodes_completed),
        collision_rate=_rate(collisions, episodes_completed),
        timeout_rate=_rate(timeouts, episodes_completed),
        final_policy_loss=final_policy_loss,
        final_value_loss=final_value_loss,
        final_entropy=final_entropy,
        checkpoint_path=str(config.checkpoint_path) if config.checkpoint_path is not None else None,
        history_path=str(config.history_path) if config.history_path is not None else None,
        phase_metrics=_summarize_phase_metrics(phase_episodes),
    )
    if config.checkpoint_path is not None:
        _save_checkpoint(config.checkpoint_path, model, optimizer, config, result)
    return result


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def _set_training_progress(env: Any, completed_timesteps: int, total_timesteps: int) -> None:
    setter = getattr(env, "set_training_progress", None)
    if setter is not None:
        setter(completed_timesteps, total_timesteps)


def _collect_rollout(
    *,
    env: Any,
    model: MLPActorCritic,
    observation: Sequence[float],
    config: PPOTrainingConfig,
    rollout_length: int,
    seed_offset: int,
    starting_timestep: int,
    current_episode_return: float,
) -> dict[str, Any]:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_returns: list[float] = []
    episodes_completed = 0
    successes = 0
    collisions = 0
    timeouts = 0
    phase_episodes: list[dict[str, Any]] = []

    for local_step in range(rollout_length):
        observations.append(_observation_tensor(observation))
        action, raw_action, logprob, value = sample_action(
            model,
            observation,
            env.settings,
            config.network,
        )
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        actions.append(raw_action)
        logprobs.append(logprob)
        values.append(value)
        rewards.append(float(reward))
        dones.append(done)
        current_episode_return += float(reward)

        if done:
            episodes_completed += 1
            completed_return = current_episode_return
            episode_returns.append(completed_return)
            current_episode_return = 0.0
            success = _episode_success(info)
            collision = bool(info.get("collided", False))
            timeout = bool(info.get("timed_out", truncated))
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            if "curriculum_phase" in info:
                phase_episodes.append(
                    {
                        "phase": str(info["curriculum_phase"]),
                        "return": completed_return,
                        "success": success,
                        "collision": collision,
                        "timeout": timeout,
                    }
                )
            completed_timesteps = starting_timestep + local_step + 1
            _set_training_progress(env, completed_timesteps, config.total_timesteps)
            next_observation, _ = env.reset(seed=config.seed + seed_offset + episodes_completed)

        observation = next_observation

    return {
        "observations": observations,
        "actions": actions,
        "logprobs": logprobs,
        "values": values,
        "rewards": rewards,
        "dones": dones,
        "observation": observation,
        "episode_returns": episode_returns,
        "episodes_completed": episodes_completed,
        "current_episode_return": current_episode_return,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "phase_episodes": phase_episodes,
    }


def _summarize_phase_metrics(records: Sequence[dict[str, Any]]) -> tuple[PPOPhaseMetrics, ...]:
    summaries = []
    for phase in sorted({str(record["phase"]) for record in records}):
        selected = [record for record in records if record["phase"] == phase]
        count = len(selected)
        summaries.append(
            PPOPhaseMetrics(
                phase=phase,
                episodes_completed=count,
                average_episode_return=_rate_sum([float(record["return"]) for record in selected]),
                success_rate=_rate(sum(bool(record["success"]) for record in selected), count),
                collision_rate=_rate(sum(bool(record["collision"]) for record in selected), count),
                timeout_rate=_rate(sum(bool(record["timeout"]) for record in selected), count),
            )
        )
    return tuple(summaries)


def _evaluate_actions(
    model: MLPActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_means, values = model(observations)
    distribution = _action_distribution(model, action_means)
    logprobs = distribution.log_prob(actions).sum(dim=-1)
    entropy = distribution.entropy().sum(dim=-1)
    return logprobs, entropy, values


def _action_distribution(model: MLPActorCritic, action_means: torch.Tensor) -> Normal:
    log_std = torch.clamp(model.log_std, min=-5.0, max=2.0)
    action_std = torch.exp(log_std).expand_as(action_means)
    return Normal(action_means, action_std)


def _raw_action_to_drone_action(
    raw_action: torch.Tensor,
    settings: SimpleAvoidanceSettings,
    network_config: MLPActorCriticConfig,
) -> DroneAction:
    min_speed_fraction = float(getattr(network_config, "min_speed_fraction", 0.0))
    speed_fraction = min_speed_fraction + torch.sigmoid(raw_action[0]).item() * (1.0 - min_speed_fraction)
    return DroneAction(
        speed=float(speed_fraction * settings.max_speed),
        heading_delta=float(torch.tanh(raw_action[1]).item() * network_config.max_heading_delta),
        climb_rate=float(torch.tanh(raw_action[2]).item() * settings.max_climb_rate),
    )


def _observation_tensor(observation: Sequence[float]) -> torch.Tensor:
    if len(observation) != 15:
        raise ValueError("observation must contain exactly 15 values")
    return torch.tensor(tuple(float(value) for value in observation), dtype=torch.float32)


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(unbiased=False)
    if std <= 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def _episode_success(info: dict[str, Any]) -> bool:
    metrics = info.get("episode_metrics")
    if metrics is not None:
        return bool(metrics.success)
    return bool(info.get("reached_goal", False)) and not bool(info.get("collided", False)) and not bool(
        info.get("timed_out", False)
    )


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(count) / float(total)


def _rate_sum(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _prepare_jsonl(path: str | Path) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")


def _append_history(path: str | Path, record: dict[str, Any]) -> None:
    _assert_json_safe(record)
    serialized = json.dumps(record, allow_nan=False, sort_keys=True)
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{serialized}\n")


def _save_checkpoint(
    path: str | Path,
    model: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOTrainingConfig,
    result: PPOTrainingResult,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "record_type": "ppo_checkpoint",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
            },
            "network_config": _json_ready(asdict(config.network)),
            "training_config": _json_ready(asdict(config)),
            "result": _json_ready(asdict(result)),
        },
        checkpoint_path,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _assert_json_safe(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("history contains a non-finite float")
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe(item)
