import importlib
import math
import sys

import pytest

from swift.rl import APFConfig
from swift.rl.apf import apf_features_from_observation, attractive_force, repulsive_force


def _observation(
    *,
    relative_goal=(3.0, 4.0, 0.0),
    relative_obstacle=(0.0, 0.0, 0.0),
    obstacle_radius=0.0,
    goal_distance=5.0,
) -> tuple[float, ...]:
    return (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *relative_goal,
        *relative_obstacle,
        obstacle_radius,
        goal_distance,
    )


def test_attractive_force_points_to_goal_with_bounded_gain():
    force = attractive_force((3.0, 4.0, 0.0), APFConfig(attractive_gain=2.0))

    assert force == pytest.approx((1.2, 1.6, 0.0))


def test_repulsive_force_is_zero_outside_influence_radius():
    force = repulsive_force(
        relative_obstacle=(5.0, 0.0, 0.0),
        obstacle_radius=0.5,
        config=APFConfig(influence_radius=2.0),
    )

    assert force == pytest.approx((0.0, 0.0, 0.0))


def test_repulsive_force_points_away_from_near_obstacle_and_is_clamped():
    force = repulsive_force(
        relative_obstacle=(0.25, 0.0, 0.0),
        obstacle_radius=0.5,
        config=APFConfig(repulsive_gain=1.0, influence_radius=2.0, max_repulsive_magnitude=3.0),
    )

    assert force == pytest.approx((-3.0, 0.0, 0.0))


def test_apf_features_from_observation_returns_strict_nine_dim_vector():
    features = apf_features_from_observation(
        _observation(relative_goal=(3.0, 4.0, 0.0), relative_obstacle=(1.0, 0.0, 0.0), obstacle_radius=0.5),
        APFConfig(attractive_gain=1.0, repulsive_gain=1.0, influence_radius=2.0),
    )

    assert features.attractive == pytest.approx((0.6, 0.8, 0.0))
    assert features.repulsive[0] < 0.0
    assert features.combined[0] == pytest.approx(features.attractive[0] + features.repulsive[0])
    assert len(features.as_tuple()) == 9
    assert all(math.isfinite(value) for value in features.as_tuple())


def test_apf_rejects_invalid_config_and_observation_length():
    with pytest.raises(ValueError, match="influence_radius"):
        APFConfig(influence_radius=0.0)
    with pytest.raises(ValueError, match="15"):
        apf_features_from_observation((0.0,) * 14)


def test_public_apf_import_does_not_load_torch_backend():
    sys.modules.pop("swift.rl", None)
    sys.modules.pop("swift.rl.torch_ppo", None)
    sys.modules.pop("swift.rl.torch_hca_ppo", None)

    rl = importlib.import_module("swift.rl")
    features = rl.apf_features_from_observation(_observation())

    assert len(features.as_tuple()) == 9
    assert "swift.rl.torch_ppo" not in sys.modules
    assert "swift.rl.torch_hca_ppo" not in sys.modules
