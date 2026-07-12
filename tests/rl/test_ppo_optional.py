import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def test_ppo_facade_does_not_import_torch_backend_eagerly():
    sys.modules.pop("swift.rl.torch_ppo", None)

    ppo = importlib.reload(importlib.import_module("swift.rl.ppo"))

    assert "swift.rl.torch_ppo" not in sys.modules
    assert ppo.PPOConfig().clip_range == 0.2


def test_training_config_defaults_are_smoke_sized_and_validated():
    from swift.rl.ppo import MLPActorCriticConfig, PPOTrainingConfig

    network = MLPActorCriticConfig()
    config = PPOTrainingConfig()
    artifact_config = PPOTrainingConfig(checkpoint_path="checkpoints/model.ckpt", history_path="outputs/history.jsonl")

    assert network.observation_dim == 15
    assert network.action_dim == 3
    assert MLPActorCriticConfig(observation_dim=27).observation_dim == 27
    assert config.total_timesteps == 128
    assert config.rollout_steps == 32
    assert config.minibatch_size == 16
    assert config.update_epochs == 1
    assert artifact_config.checkpoint_path == Path("checkpoints/model.ckpt")
    assert artifact_config.history_path == Path("outputs/history.jsonl")
    assert "swift.rl.torch_ppo" not in sys.modules
    with pytest.raises(ValueError, match="total_timesteps"):
        PPOTrainingConfig(total_timesteps=0)
    with pytest.raises(ValueError, match="minibatch_size"):
        PPOTrainingConfig(rollout_steps=8, minibatch_size=16)
    with pytest.raises(ValueError, match="observation_dim"):
        MLPActorCriticConfig(observation_dim=14)


def test_mlp_config_preserves_visibility_fields_from_compatible_apf_object():
    from swift.rl.ppo import MLPActorCriticConfig

    prior = SimpleNamespace(
        attractive_gain=1.0,
        repulsive_gain=0.02,
        influence_radius=0.4,
        max_repulsive_magnitude=10.0,
        epsilon=1e-6,
        ignore_obstacles_behind=True,
        bypass_enabled=False,
        bypass_lateral_offset=0.25,
        bypass_forward_margin=0.08,
        bypass_clearance=0.16,
        visibility_planner_enabled=True,
        visibility_clearance=0.18,
        visibility_samples=16,
    )

    config = MLPActorCriticConfig(apf_action_prior=prior)

    assert config.apf_action_prior is not None
    assert config.apf_action_prior.visibility_planner_enabled is True
    assert config.apf_action_prior.visibility_clearance == pytest.approx(0.18)
    assert config.apf_action_prior.visibility_samples == 16


def test_train_ppo_mlp_raises_domain_error_when_torch_is_unavailable(monkeypatch):
    from swift.rl import ppo

    def missing_torch_backend(name, package=None):
        if name == ".torch_ppo" and package == "swift.rl":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return importlib.import_module(name, package)

    monkeypatch.setattr(ppo, "import_module", missing_torch_backend)

    with pytest.raises(ppo.TorchUnavailableError, match="torch"):
        ppo.train_ppo_mlp(lambda: None)
