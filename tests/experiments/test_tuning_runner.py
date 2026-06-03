import json
import math

import pytest

from swift.core import ObstacleState
from swift.experiments.tuning_runner import (
    PolicySearchSpace,
    Stage1Scenario,
    TuningRunConfig,
    evaluate_policy,
    run_stage1_policy_search,
)
from swift.rl import MLPBaselinePolicyConfig


def test_default_baseline_collides_in_fixed_stage1_scenario():
    settings = Stage1Scenario().build_settings()

    metrics = evaluate_policy(settings, MLPBaselinePolicyConfig())

    assert metrics["success"] is False
    assert metrics["collided"] is True
    assert metrics["timed_out"] is False
    assert metrics["minimum_safety_distance"] < settings.safety_margin


def test_stage1_policy_search_finds_accepted_successful_candidate():
    result = run_stage1_policy_search()

    assert result["acceptance"]["accepted"] is True
    assert result["acceptance"]["accepted_candidate_count"] >= 1
    assert len(result["candidates"]) == result["acceptance"]["candidate_count"]
    assert result["baseline"]["collided"] is True
    assert result["best_candidate"]["accepted"] is True
    assert result["best_candidate"]["metrics"]["success"] is True
    assert result["best_candidate"]["metrics"]["collided"] is False
    assert result["best_candidate"]["metrics"]["timed_out"] is False
    assert result["best_candidate"]["metrics"]["minimum_safety_distance"] >= result["acceptance"]["minimum_safety_distance"]
    assert len(result["baseline"]["apf"]["combined"]) == 3
    assert result["baseline"]["apf"]["combined_norm"] > 0.0
    assert len(result["best_candidate"]["apf"]["as_tuple"]) == 9
    assert all(math.isfinite(value) for value in result["best_candidate"]["apf"]["as_tuple"])
    assert result["best_candidate"]["config"]

    expected_config = {
        "max_speed": 0.8,
        "max_heading_delta": 0.6,
        "max_climb_rate": 0.5,
        "obstacle_avoidance_distance": 4.0,
        "avoidance_heading_delta": 0.4,
    }
    assert any(candidate["config"] == expected_config and candidate["accepted"] for candidate in result["candidates"])
    json.dumps(result, allow_nan=False, sort_keys=True)


def test_stage1_policy_search_can_limit_candidate_output():
    config = TuningRunConfig(
        search_space=PolicySearchSpace(),
        candidate_limit=2,
    )

    result = run_stage1_policy_search(config)

    assert result["acceptance"]["candidate_count"] == 8
    assert len(result["candidates"]) == 2
    assert result["candidates"][0] == result["best_candidate"]


def test_best_candidate_config_keeps_no_obstacle_baseline_successful():
    result = run_stage1_policy_search()
    best_config = MLPBaselinePolicyConfig(**result["best_candidate"]["config"])
    no_obstacle_settings = Stage1Scenario(obstacles=()).build_settings()

    metrics = evaluate_policy(no_obstacle_settings, best_config)

    assert metrics["success"] is True
    assert metrics["collided"] is False
    assert metrics["timed_out"] is False
    assert metrics["minimum_safety_distance"] == 0.0


def test_stage1_scenario_builds_fixed_simple_avoidance_settings():
    obstacle = ObstacleState(position=(4.0, 0.0, 0.0), radius=0.6)

    settings = Stage1Scenario().build_settings()

    assert settings.start == (0.0, 0.0, 0.0)
    assert settings.goal == (10.0, 0.0, 0.0)
    assert settings.obstacles == (obstacle,)
    assert settings.max_steps == 60
    assert settings.goal_radius == 0.5
    assert settings.safety_margin == 0.25


def test_tuning_config_rejects_non_finite_safety_threshold():
    with pytest.raises(ValueError, match="finite"):
        TuningRunConfig(minimum_safety_distance=math.nan)


def test_policy_search_space_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        PolicySearchSpace(max_speed=(math.inf,))
