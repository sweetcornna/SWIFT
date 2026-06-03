import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from swift.core import DroneAction
from swift.config import SimulationSettings
from swift.envs import PyBulletVelocityTrainingEnv, SimpleAvoidanceEnv, SimpleAvoidanceSettings
from swift.rl.ppo import MLPActorCriticConfig, PPOTrainingConfig
from swift.rl.torch_ppo import (
    MLPActorCritic,
    _raw_action_to_drone_action,
    compute_gae,
    deterministic_action,
    load_ppo_mlp_checkpoint,
    sample_action,
    train_ppo_mlp,
)


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


def test_raw_action_to_drone_action_applies_minimum_speed_fraction():
    settings = SimpleAvoidanceSettings(max_speed=2.0, max_climb_rate=0.4)
    config = MLPActorCriticConfig(
        hidden_sizes=(8,),
        max_heading_delta=0.35,
        min_speed_fraction=0.4,
    )

    action = _raw_action_to_drone_action(torch.tensor([0.0, 0.0, 0.0]), settings, config)

    assert action.speed == pytest.approx(1.4)
    assert action.heading_delta == pytest.approx(0.0)
    assert action.climb_rate == pytest.approx(0.0)


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


def test_train_ppo_mlp_closes_env_after_training():
    created = []

    def make_env():
        env = ClosingEnv()
        created.append(env)
        return env

    train_ppo_mlp(
        make_env,
        PPOTrainingConfig(
            total_timesteps=32,
            rollout_steps=16,
            minibatch_size=8,
            update_epochs=1,
            torch_num_threads=1,
            seed=123,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
        ),
    )

    assert created
    assert created[0].closed is True


def test_train_ppo_mlp_runs_tiny_update_with_pybullet_training_env_fake_runtime(tmp_path: Path):
    def make_env():
        return PyBulletVelocityTrainingEnv(
            simulation_settings=SimulationSettings(
                pybullet_root=tmp_path,
                pixi_executable=tmp_path / "pixi.exe",
                required_tasks=(),
                check_task="test",
                smoke_task="drone-demo",
                command_timeout_seconds=30,
            ),
            settings=SimpleAvoidanceSettings(max_steps=4, max_speed=1.0, max_climb_rate=0.5),
            velocity_aviary_cls=lambda **_: FakeVelocityAviary(),
            drone_model=SimpleNamespace(CF2X="cf2x"),
            physics=SimpleNamespace(PYB="pyb"),
        )

    result = train_ppo_mlp(
        make_env,
        PPOTrainingConfig(
            total_timesteps=32,
            rollout_steps=16,
            minibatch_size=8,
            update_epochs=1,
            torch_num_threads=1,
            seed=123,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
        ),
    )

    assert result.total_timesteps == 32
    assert result.updates >= 1
    assert result.episodes_completed >= 1
    assert math.isfinite(result.average_episode_return)


def test_train_ppo_mlp_writes_checkpoint_and_update_history(tmp_path: Path):
    def make_env():
        return SimpleAvoidanceEnv(SimpleAvoidanceSettings(max_steps=8, goal=(4.0, 0.0, 0.0)))

    checkpoint_path = tmp_path / "checkpoints" / "ppo.pt"
    history_path = tmp_path / "history" / "updates.jsonl"

    result = train_ppo_mlp(
        make_env,
        PPOTrainingConfig(
            total_timesteps=64,
            rollout_steps=32,
            minibatch_size=16,
            update_epochs=1,
            torch_num_threads=1,
            seed=123,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
            checkpoint_path=checkpoint_path,
            history_path=history_path,
        ),
    )

    assert result.checkpoint_path == str(checkpoint_path)
    assert result.history_path == str(history_path)

    raw_history = history_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_history
    assert "Infinity" not in raw_history
    assert raw_history.endswith("\n")
    history_lines = raw_history.splitlines()
    history = [json.loads(line) for line in history_lines]
    assert len(history) == result.updates
    assert [record["update"] for record in history] == list(range(1, result.updates + 1))
    for line, record in zip(history_lines, history, strict=True):
        assert json.dumps(record, allow_nan=False, sort_keys=True) == line
        assert record["record_type"] == "ppo_update"
        assert record["total_timesteps"] > 0
        assert record["episodes_completed"] >= 0
        for metric in ("policy_loss", "value_loss", "entropy", "average_episode_return"):
            assert math.isfinite(record[metric])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert {
        "schema_version",
        "record_type",
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "network_config",
        "training_config",
        "result",
    } <= set(checkpoint)
    assert checkpoint["schema_version"] == 1
    assert checkpoint["record_type"] == "ppo_checkpoint"
    assert checkpoint["training_config"]["total_timesteps"] == 64
    assert checkpoint["training_config"]["network"]["hidden_sizes"] == [8]
    assert checkpoint["training_config"]["checkpoint_path"] == str(checkpoint_path)
    assert checkpoint["training_config"]["history_path"] == str(history_path)
    assert checkpoint["result"]["updates"] == result.updates
    assert "actor_head.weight" in checkpoint["model_state_dict"]
    assert "state" in checkpoint["optimizer_state_dict"]
    assert "torch_cpu" in checkpoint["rng_state"]
    assert isinstance(checkpoint["rng_state"]["python"], tuple)


def test_load_checkpoint_reconstructs_model_for_deterministic_action(tmp_path: Path):
    def make_env():
        return SimpleAvoidanceEnv(SimpleAvoidanceSettings(max_steps=8, goal=(4.0, 0.0, 0.0)))

    checkpoint_path = tmp_path / "ppo.ckpt"
    train_ppo_mlp(
        make_env,
        PPOTrainingConfig(
            total_timesteps=64,
            rollout_steps=32,
            minibatch_size=16,
            update_epochs=1,
            torch_num_threads=1,
            seed=123,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
            checkpoint_path=checkpoint_path,
        ),
    )
    env = make_env()
    observation, _ = env.reset(seed=123)

    model, checkpoint = load_ppo_mlp_checkpoint(checkpoint_path)
    action = deterministic_action(model, observation, env.settings, model.config)

    assert checkpoint["record_type"] == "ppo_checkpoint"
    assert model.config.hidden_sizes == (8,)
    assert 0.0 <= action.speed <= env.settings.max_speed
    assert -model.config.max_heading_delta <= action.heading_delta <= model.config.max_heading_delta
    assert -env.settings.max_climb_rate <= action.climb_rate <= env.settings.max_climb_rate


def test_load_checkpoint_rejects_wrong_record_type(tmp_path: Path):
    checkpoint_path = tmp_path / "bad.ckpt"
    torch.save({"record_type": "not_ppo_checkpoint"}, checkpoint_path)

    with pytest.raises(ValueError, match="ppo_checkpoint"):
        load_ppo_mlp_checkpoint(checkpoint_path)


class ClosingEnv:
    settings = SimpleAvoidanceSettings(max_steps=1)

    def __init__(self) -> None:
        self.closed = False

    def reset(self, seed=None):
        return (0.0,) * 15, {"seed": seed}

    def step(self, action):
        return (0.0,) * 15, -1.0, False, True, {"timed_out": True, "collided": False}

    def close(self):
        self.closed = True


class FakeVelocityAviary:
    def reset(self, seed=None, options=None):
        return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]], {
            "seed": seed
        }

    def step(self, action):
        return (
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]],
            -1.0,
            False,
            False,
            {"source": "fake"},
        )

    def close(self):
        pass
