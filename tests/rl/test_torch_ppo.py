import math

import pytest

torch = pytest.importorskip("torch")

from swift.core import DroneAction
from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings
from swift.rl.ppo import MLPActorCriticConfig, PPOTrainingConfig
from swift.rl.torch_ppo import MLPActorCritic, compute_gae, sample_action, train_ppo_mlp


def test_actor_critic_outputs_actor_and_value_shapes():
    config = MLPActorCriticConfig(hidden_sizes=(8,))
    model = MLPActorCritic(config)

    means, values = model(torch.zeros((2, config.observation_dim), dtype=torch.float32))

    assert means.shape == (2, config.action_dim)
    assert values.shape == (2,)
    assert model.log_std.shape == (config.action_dim,)


def test_sample_action_returns_bounded_drone_action_and_training_tensors():
    settings = SimpleAvoidanceSettings(max_speed=1.25, max_climb_rate=0.4)
    env = SimpleAvoidanceEnv(settings)
    observation, _ = env.reset()
    config = MLPActorCriticConfig(hidden_sizes=(8,), max_heading_delta=0.35)
    model = MLPActorCritic(config)

    action, raw_action, logprob, value = sample_action(model, observation, settings, config)

    assert isinstance(action, DroneAction)
    assert 0.0 <= action.speed <= settings.max_speed
    assert -config.max_heading_delta <= action.heading_delta <= config.max_heading_delta
    assert -settings.max_climb_rate <= action.climb_rate <= settings.max_climb_rate
    assert raw_action.shape == (config.action_dim,)
    assert logprob.shape == ()
    assert value.shape == ()


def test_compute_gae_returns_finite_returns_and_advantages():
    rewards = torch.tensor([1.0, 0.5, -1.0], dtype=torch.float32)
    values = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    dones = torch.tensor([False, True, False])
    next_value = torch.tensor(0.4, dtype=torch.float32)

    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert returns.shape == rewards.shape
    assert advantages.shape == rewards.shape
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()
    assert returns[1].item() == pytest.approx(rewards[1].item())


def test_train_ppo_mlp_runs_tiny_cpu_update_and_reports_finite_metrics():
    def make_env():
        return SimpleAvoidanceEnv(SimpleAvoidanceSettings(max_steps=8, goal=(4.0, 0.0, 0.0)))

    result = train_ppo_mlp(
        make_env,
        PPOTrainingConfig(
            total_timesteps=128,
            rollout_steps=32,
            minibatch_size=16,
            update_epochs=1,
            torch_num_threads=1,
            seed=123,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
        ),
    )

    assert result.total_timesteps == 128
    assert result.updates >= 1
    assert result.episodes_completed >= 1
    for metric in (
        result.average_episode_return,
        result.success_rate,
        result.collision_rate,
        result.timeout_rate,
        result.final_policy_loss,
        result.final_value_loss,
        result.final_entropy,
    ):
        assert math.isfinite(metric)
