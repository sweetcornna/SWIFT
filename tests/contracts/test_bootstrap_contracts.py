from pathlib import Path
import tomllib

import pytest

from swift.envs import (
    BootstrapDroneEnv,
    SimpleAvoidanceEnv,
    SimpleAvoidanceSettings,
    UnsupportedOperationError,
)
from swift.experiments import ExperimentMetric, ExperimentSpec
from swift.rl import APFConfig, HCAConfig, PPOConfig


def test_bootstrap_environment_declares_shapes_and_blocks_runtime_use():
    env = BootstrapDroneEnv(observation_size=24, action_size=3)

    assert env.observation_shape == (24,)
    assert env.action_shape == (3,)
    with pytest.raises(UnsupportedOperationError, match="outside bootstrap scope"):
        env.reset()


def test_envs_export_bootstrap_and_simple_avoidance_contracts():
    env = SimpleAvoidanceEnv(SimpleAvoidanceSettings())

    assert BootstrapDroneEnv(observation_size=24, action_size=3).observation_shape == (24,)
    assert env.observation_shape == (15,)
    assert env.action_shape == (3,)


def test_rl_configs_capture_project_defaults():
    assert PPOConfig().clip_range == 0.2
    assert HCAConfig().target_attention_heads == 4
    assert APFConfig().repulsive_gain == 1.0


def test_experiment_spec_names_ablation_variants():
    spec = ExperimentSpec(
        name="bootstrap-ablation-matrix",
        variants=("mlp_ppo", "hca_ppo", "hca_apf_ppo"),
        metrics=(
            ExperimentMetric.SUCCESS_RATE,
            ExperimentMetric.PATH_SMOOTHNESS,
        ),
    )

    assert spec.variants == ("mlp_ppo", "hca_ppo", "hca_apf_ppo")


def test_pyproject_declares_sim_extra_for_optional_pybullet_runtime():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    sim_extra = pyproject["project"]["optional-dependencies"]["sim"]

    assert any(dependency.startswith("gymnasium") for dependency in sim_extra)
    assert any(dependency.startswith("pybullet") for dependency in sim_extra)
    assert any(dependency.startswith("numpy") for dependency in sim_extra)


def test_training_and_sim_extras_allow_pybullet_pixi_numpy_contract():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    train_numpy = [dependency for dependency in optional_dependencies["train"] if dependency.startswith("numpy")]
    sim_numpy = [dependency for dependency in optional_dependencies["sim"] if dependency.startswith("numpy")]

    assert train_numpy == ["numpy>=1.26,<3"]
    assert sim_numpy == ["numpy>=1.26,<3"]
