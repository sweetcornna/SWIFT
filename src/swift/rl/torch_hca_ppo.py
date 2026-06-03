from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from swift.core import DroneAction
from swift.envs import SimpleAvoidanceSettings
from swift.rl.apf import APFConfig
from swift.rl.hca import HCAActorCriticConfig, HCAObservationAdapterConfig
from swift.rl.ppo import HCAPPOTrainingConfig, PPOTrainingResult
from swift.rl.torch_ppo import (
    _action_distribution,
    _append_history,
    _episode_success,
    _evaluate_actions,
    _json_ready,
    _normalize_advantages,
    _observation_tensor,
    _prepare_jsonl,
    _rate,
    _rate_sum,
    _raw_action_to_drone_action,
    compute_gae,
)


@dataclass(frozen=True)
class HCATokens:
    self_state: torch.Tensor
    target_state: torch.Tensor
    threat_state: torch.Tensor


class HCAObservationAdapter(nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = config or HCAActorCriticConfig().observation

    def forward(self, observations: torch.Tensor) -> HCATokens:
        observations = observations.to(dtype=torch.float32)
        if observations.ndim != 2 or observations.shape[1] != self.config.observation_dim:
            raise ValueError(f"observations must have shape (batch, {self.config.observation_dim})")
        return HCATokens(
            self_state=observations[:, 0:7],
            target_state=torch.cat((observations[:, 7:10], observations[:, 14:15]), dim=1),
            threat_state=observations[:, 10:14],
        )


class HCAFeatureExtractor(nn.Module):
    def __init__(self, config: HCAActorCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or HCAActorCriticConfig()
        self.adapter = HCAObservationAdapter(self.config.observation)
        embedding_dim = self.config.embedding_dim
        self.self_projection = nn.Linear(self.config.observation.self_dim, embedding_dim)
        self.target_projection = nn.Linear(self.config.observation.target_dim, embedding_dim)
        self.threat_projection = nn.Linear(self.config.observation.threat_dim, embedding_dim)
        self.target_attention = nn.MultiheadAttention(
            embedding_dim,
            self.config.target_attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.threat_attention = nn.MultiheadAttention(
            embedding_dim,
            self.config.threat_attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        if self.config.apf_config is None:
            self.apf_projection = None
        else:
            self.apf_projection = nn.Linear(9, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tokens = self.adapter(observations)
        self_query = self.self_projection(tokens.self_state).unsqueeze(1)
        target_tokens = self.target_projection(tokens.target_state).unsqueeze(1)
        threat_tokens = self.threat_projection(tokens.threat_state).unsqueeze(1)
        target_context, _ = self.target_attention(self_query, target_tokens, target_tokens)
        threat_context, _ = self.threat_attention(target_context, threat_tokens, threat_tokens)
        fused = target_context + threat_context
        if self.config.apf_config is not None and self.apf_projection is not None:
            apf_context = self.apf_projection(_apf_feature_tensor(observations, self.config.apf_config)).unsqueeze(1)
            fused = fused + apf_context
        fused = self.layer_norm(fused)
        return fused.squeeze(1)


class HCAActorCritic(nn.Module):
    def __init__(self, config: HCAActorCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or HCAActorCriticConfig()
        self.feature_extractor = HCAFeatureExtractor(self.config)
        layers: list[nn.Module] = []
        input_dim = self.config.embedding_dim
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
        features = self.backbone(self.feature_extractor(observations))
        action_means = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)
        return action_means, values


def sample_hca_action(
    model: HCAActorCritic,
    observation: Sequence[float],
    settings: SimpleAvoidanceSettings,
    network_config: HCAActorCriticConfig,
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


def deterministic_hca_action(
    model: HCAActorCritic,
    observation: Sequence[float],
    settings: SimpleAvoidanceSettings,
    network_config: HCAActorCriticConfig,
) -> DroneAction:
    observation_tensor = _observation_tensor(observation)
    model.eval()
    with torch.no_grad():
        action_means, _ = model(observation_tensor.unsqueeze(0))
    return _raw_action_to_drone_action(action_means[0].detach().cpu(), settings, network_config)


def load_ppo_hca_checkpoint(path: str | Path) -> tuple[HCAActorCritic, dict[str, Any]]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("record_type") != "ppo_hca_checkpoint":
        raise ValueError("checkpoint record_type must be ppo_hca_checkpoint")
    network_config = _hca_network_config_from_checkpoint(checkpoint["network_config"])
    model = HCAActorCritic(network_config).to(torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def train_ppo_hca(
    make_env: Callable[[], Any],
    config: HCAPPOTrainingConfig,
) -> PPOTrainingResult:
    torch.set_num_threads(config.torch_num_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    model = HCAActorCritic(config.network).to(torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    env = make_env()
    observation, _ = env.reset(seed=config.seed)

    total_timesteps = 0
    updates = 0
    episodes_completed = 0
    episode_returns: list[float] = []
    current_episode_return = 0.0
    successes = 0
    collisions = 0
    timeouts = 0
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
                loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                final_policy_loss = float(policy_loss.detach().item())
                final_value_loss = float(value_loss.detach().item())
                final_entropy = float(entropy_loss.detach().item())
        updates += 1
        if config.history_path is not None:
            _append_history(
                config.history_path,
                {
                    "schema_version": 1,
                    "record_type": "ppo_hca_update",
                    "update": updates,
                    "total_timesteps": total_timesteps,
                    "episodes_completed": episodes_completed,
                    "success_rate": _rate(successes, episodes_completed),
                    "collision_rate": _rate(collisions, episodes_completed),
                    "timeout_rate": _rate(timeouts, episodes_completed),
                    "average_episode_return": _rate_sum(episode_returns),
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
    )
    if config.checkpoint_path is not None:
        _save_hca_checkpoint(config.checkpoint_path, model, optimizer, config, result)
    return result


def _apf_feature_tensor(observations: torch.Tensor, config: APFConfig) -> torch.Tensor:
    observations = observations.to(dtype=torch.float32)
    if observations.ndim != 2 or observations.shape[1] != 15:
        raise ValueError("observations must have shape (batch, 15)")

    relative_goal = observations[:, 7:10]
    relative_obstacle = observations[:, 10:13]
    obstacle_radius = observations[:, 13:14]

    attractive = _unit_vectors(relative_goal, epsilon=0.0) * float(config.attractive_gain)
    obstacle_distance = torch.linalg.vector_norm(relative_obstacle, dim=1, keepdim=True)
    default_direction = torch.zeros_like(relative_obstacle)
    default_direction[:, 0] = -1.0
    direction = torch.where(
        obstacle_distance <= float(config.epsilon),
        default_direction,
        -relative_obstacle / obstacle_distance.clamp_min(float(config.epsilon)),
    )
    safety_distance = (obstacle_distance - obstacle_radius).clamp_min(float(config.epsilon))
    magnitude = float(config.repulsive_gain) * (
        (1.0 / safety_distance) - (1.0 / float(config.influence_radius))
    ) / (safety_distance * safety_distance)
    magnitude = magnitude.clamp_max(float(config.max_repulsive_magnitude))
    active = (obstacle_radius > 0.0) & (safety_distance < float(config.influence_radius))
    repulsive = direction * torch.where(active, magnitude, torch.zeros_like(magnitude))
    combined = attractive + repulsive
    return torch.cat((attractive, repulsive, combined), dim=1)


def _hca_network_config_from_checkpoint(payload: dict[str, Any]) -> HCAActorCriticConfig:
    observation_payload = payload.get("observation")
    observation = (
        HCAObservationAdapterConfig(**observation_payload)
        if isinstance(observation_payload, dict)
        else HCAObservationAdapterConfig()
    )
    apf_payload = payload.get("apf_config")
    apf_config = APFConfig(**apf_payload) if isinstance(apf_payload, dict) else None
    return HCAActorCriticConfig(
        observation=observation,
        action_dim=int(payload.get("action_dim", 3)),
        embedding_dim=int(payload.get("embedding_dim", 64)),
        target_attention_heads=int(payload.get("target_attention_heads", 4)),
        threat_attention_heads=int(payload.get("threat_attention_heads", 4)),
        hidden_sizes=tuple(int(size) for size in payload.get("hidden_sizes", (64,))),
        dropout=float(payload.get("dropout", 0.0)),
        max_heading_delta=float(payload.get("max_heading_delta", 0.5)),
        log_std_init=float(payload.get("log_std_init", -0.5)),
        apf_config=apf_config,
    )


def _unit_vectors(vectors: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    return torch.where(norm > epsilon, vectors / norm.clamp_min(max(epsilon, 1e-12)), torch.zeros_like(vectors))


def _collect_rollout(
    *,
    env: Any,
    model: HCAActorCritic,
    observation: Sequence[float],
    config: HCAPPOTrainingConfig,
    rollout_length: int,
    seed_offset: int,
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

    for _ in range(rollout_length):
        observations.append(_observation_tensor(observation))
        action, raw_action, logprob, value = sample_hca_action(
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
            episode_returns.append(current_episode_return)
            current_episode_return = 0.0
            successes += int(_episode_success(info))
            collisions += int(bool(info.get("collided", False)))
            timeouts += int(bool(info.get("timed_out", truncated)))
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
    }


def _save_hca_checkpoint(
    path: str | Path,
    model: HCAActorCritic,
    optimizer: torch.optim.Optimizer,
    config: HCAPPOTrainingConfig,
    result: PPOTrainingResult,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "record_type": "ppo_hca_checkpoint",
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
