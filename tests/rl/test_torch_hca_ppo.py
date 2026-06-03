import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from swift.core import DroneAction
from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings
from swift.rl import APFConfig
from swift.rl.hca import HCAActorCriticConfig
from swift.rl.ppo import HCAPPOTrainingConfig
from swift.rl.torch_hca_ppo import (
    HCAActorCritic,
    HCAFeatureExtractor,
    HCAObservationAdapter,
    sample_hca_action,
    train_ppo_hca,
)


def _observation_tensor():
    return torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                0.1,
                0.2,
                0.3,
                0.4,
                4.0,
                5.0,
                6.0,
                -1.0,
                -2.0,
                -3.0,
                0.5,
                8.0,
            ]
        ],
        dtype=torch.float32,
    )


def test_hca_observation_adapter_splits_15d_observation_into_tokens():
    adapter = HCAObservationAdapter()

    tokens = adapter(_observation_tensor())

    assert tokens.self_state.shape == (1, 7)
    assert tokens.target_state.shape == (1, 4)
    assert tokens.threat_state.shape == (1, 4)
    assert tokens.self_state[0].tolist() == pytest.approx([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4])
    assert tokens.target_state[0].tolist() == pytest.approx([4.0, 5.0, 6.0, 8.0])
    assert tokens.threat_state[0].tolist() == pytest.approx([-1.0, -2.0, -3.0, 0.5])


def test_hca_feature_extractor_returns_finite_embedding():
    config = HCAActorCriticConfig(embedding_dim=16, target_attention_heads=4, threat_attention_heads=4)
    extractor = HCAFeatureExtractor(config)

    features = extractor(_observation_tensor())

    assert features.shape == (1, 16)
    assert torch.isfinite(features).all()


def test_hca_feature_extractor_fuses_apf_without_expanding_observation():
    config = HCAActorCriticConfig(
        embedding_dim=16,
        target_attention_heads=4,
        threat_attention_heads=4,
        apf_config=APFConfig(attractive_gain=0.5, repulsive_gain=0.25),
    )
    extractor = HCAFeatureExtractor(config)

    features = extractor(_observation_tensor())

    assert features.shape == (1, 16)
    assert torch.isfinite(features).all()
    assert extractor.apf_projection.in_features == 9
    with pytest.raises(ValueError, match="shape"):
        extractor(torch.zeros((1, 24), dtype=torch.float32))


def test_hca_actor_critic_outputs_actor_value_and_log_std_shapes():
    config = HCAActorCriticConfig(embedding_dim=16, hidden_sizes=(8,))
    model = HCAActorCritic(config)

    means, values = model(torch.zeros((2, 15), dtype=torch.float32))

    assert means.shape == (2, 3)
    assert values.shape == (2,)
    assert model.log_std.shape == (3,)


def test_sample_hca_action_returns_bounded_drone_action():
    settings = SimpleAvoidanceSettings(max_speed=1.25, max_climb_rate=0.4)
    env = SimpleAvoidanceEnv(settings)
    observation, _ = env.reset()
    config = HCAActorCriticConfig(embedding_dim=16, hidden_sizes=(8,), max_heading_delta=0.35)
    model = HCAActorCritic(config)

    action, raw_action, logprob, value = sample_hca_action(model, observation, settings, config)

    assert isinstance(action, DroneAction)
    assert 0.0 <= action.speed <= settings.max_speed
    assert -config.max_heading_delta <= action.heading_delta <= config.max_heading_delta
    assert -settings.max_climb_rate <= action.climb_rate <= settings.max_climb_rate
    assert raw_action.shape == (config.action_dim,)
    assert logprob.shape == ()
    assert value.shape == ()


def test_train_ppo_hca_runs_tiny_cpu_update_and_writes_artifacts(tmp_path: Path):
    def make_env():
        return SimpleAvoidanceEnv(SimpleAvoidanceSettings(max_steps=8, goal=(4.0, 0.0, 0.0)))

    checkpoint_path = tmp_path / "hca.ckpt"
    history_path = tmp_path / "hca.jsonl"
    result = train_ppo_hca(
        make_env,
        HCAPPOTrainingConfig(
            total_timesteps=64,
            rollout_steps=32,
            minibatch_size=16,
            update_epochs=1,
            torch_num_threads=1,
            seed=123,
            network=HCAActorCriticConfig(embedding_dim=16, hidden_sizes=(8,)),
            checkpoint_path=checkpoint_path,
            history_path=history_path,
        ),
    )

    assert result.total_timesteps == 64
    assert result.updates >= 1
    assert result.checkpoint_path == str(checkpoint_path)
    assert result.history_path == str(history_path)
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

    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert {record["record_type"] for record in history} == {"ppo_hca_update"}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["record_type"] == "ppo_hca_checkpoint"
    assert checkpoint["network_config"]["embedding_dim"] == 16
