import importlib
from pathlib import Path
import sys

import pytest


def test_hca_public_configs_do_not_import_torch_backend_eagerly():
    sys.modules.pop("swift.rl.torch_hca_ppo", None)

    hca = importlib.reload(importlib.import_module("swift.rl.hca"))

    config = hca.HCAActorCriticConfig(embedding_dim=16, hidden_sizes=(8,))
    assert config.observation.observation_dim == 15
    assert config.action_dim == 3
    assert "swift.rl.torch_hca_ppo" not in sys.modules


def test_hca_actor_critic_config_validates_shapes_heads_and_dropout():
    from swift.rl.hca import HCAActorCriticConfig, HCAObservationAdapterConfig

    observation = HCAObservationAdapterConfig()
    config = HCAActorCriticConfig(
        observation=observation,
        embedding_dim=16,
        target_attention_heads=4,
        threat_attention_heads=4,
        hidden_sizes=(8,),
        dropout=0.0,
    )

    assert observation.self_dim == 7
    assert observation.target_dim == 4
    assert observation.threat_dim == 4
    assert config.embedding_dim == 16
    with pytest.raises(ValueError, match="observation_dim"):
        HCAObservationAdapterConfig(observation_dim=14)
    with pytest.raises(ValueError, match="target_attention_heads"):
        HCAActorCriticConfig(embedding_dim=10, target_attention_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        HCAActorCriticConfig(dropout=1.5)


def test_hca_ppo_training_config_defaults_and_paths_are_torch_lazy():
    from swift.rl.ppo import HCAPPOTrainingConfig

    config = HCAPPOTrainingConfig(checkpoint_path="checkpoints/hca.ckpt", history_path="outputs/hca.jsonl")

    assert config.total_timesteps == 128
    assert config.rollout_steps == 32
    assert config.minibatch_size == 16
    assert config.checkpoint_path == Path("checkpoints/hca.ckpt")
    assert config.history_path == Path("outputs/hca.jsonl")
    assert "swift.rl.torch_hca_ppo" not in sys.modules


def test_train_ppo_hca_raises_domain_error_when_torch_is_unavailable(monkeypatch):
    from swift.rl import ppo

    def missing_torch_backend(name, package=None):
        if name == ".torch_hca_ppo" and package == "swift.rl":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return importlib.import_module(name, package)

    monkeypatch.setattr(ppo, "import_module", missing_torch_backend)

    with pytest.raises(ppo.TorchUnavailableError, match="torch"):
        ppo.train_ppo_hca(lambda: None)
