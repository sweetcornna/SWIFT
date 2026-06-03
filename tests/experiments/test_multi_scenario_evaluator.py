import json
from pathlib import Path

import pytest

from swift.core import ObstacleState
from swift.experiments.multi_scenario_evaluator import (
    MultiScenarioEvaluationConfig,
    ScenarioCase,
    run_multi_scenario_evaluation,
)
from swift.experiments.tuning_runner import PolicySearchSpace, Stage1Scenario


def test_multi_scenario_evaluation_writes_ranked_stage4_report_and_manifest(tmp_path: Path):
    output = tmp_path / "multi_scenario.json"
    config = MultiScenarioEvaluationConfig(
        scenarios=(
            ScenarioCase(identifier="fixed-obstacle", scenario=Stage1Scenario()),
            ScenarioCase(identifier="open-corridor", scenario=Stage1Scenario(obstacles=(), max_steps=30)),
        ),
        search_space=PolicySearchSpace(
            max_speed=(1.0, 0.8),
            max_heading_delta=(0.5, 0.6),
            obstacle_avoidance_distance=(2.0, 4.0),
            avoidance_heading_delta=(0.4,),
            max_climb_rate=(0.5,),
        ),
        output=output,
    )

    report = run_multi_scenario_evaluation(config)

    raw_report = output.read_text(encoding="utf-8")
    assert json.loads(raw_report) == report
    assert report["schema_version"] == 1
    assert report["record_type"] == "multi_scenario_evaluation_report"
    assert report["stage"] == "stage4"
    assert report["variant"] == "multi_scenario_policy_tuning"
    assert report["report_scope"] == "stage4_multi_scenario_evaluation"
    assert report["evaluation_level"] == "deterministic_multi_scenario_tuning"
    assert report["readiness"]["all_scenarios_evaluated"] is True
    assert report["readiness"]["convergence_claim"] is False
    assert report["readiness"]["evidence_level"] == "deterministic_multi_scenario_tuning"
    assert report["summary"]["scenario_count"] == 2
    assert report["summary"]["candidate_count"] == 8
    assert report["summary"]["globally_accepted_candidate_count"] >= 1
    assert report["summary"]["best_config"] == report["ranking"][0]["config"]
    assert report["summary"]["best_success_count"] >= 1
    assert report["summary"]["baseline_collision_count"] >= 0
    assert report["ranking_formula"] == {
        "sort": (
            "accepted_all_desc_success_rate_desc_collision_rate_asc_timeout_rate_asc_"
            "average_path_length_asc_average_path_smoothness_asc_minimum_safety_distance_desc_config_asc"
        ),
        "convergence_claim": False,
    }
    assert report["ranking"][0]["accepted_all"] is True
    assert report["ranking"][0]["metrics"]["success_rate"] == 1.0
    assert len(report["ranking"][0]["scenarios"]) == 2
    assert [scenario["scenario_id"] for scenario in report["scenarios"]] == ["fixed-obstacle", "open-corridor"]
    assert all(scenario["acceptance"]["candidate_count"] == 8 for scenario in report["scenarios"])
    assert all(scenario["acceptance"]["accepted"] is True for scenario in report["scenarios"])
    assert all("best_candidate" in scenario for scenario in report["scenarios"])
    assert all(
        set(scenario["scenario_matrix"]) == {
            "building_density",
            "dynamic_drone_count",
            "delivery_target_set",
            "altitude_profile",
        }
        for scenario in report["scenarios"]
    )
    assert all(
        set(scenario["run_provenance"]) >= {
            "seed",
            "episode_count",
            "max_steps",
            "environment_adapter",
            "policy_family",
            "minimum_safety_distance",
        }
        for scenario in report["scenarios"]
    )
    assert all(
        set(scenario["metrics"]) >= {
            "success",
            "collided",
            "timed_out",
            "path_length",
            "path_smoothness",
            "minimum_safety_distance",
        }
        for scenario in report["scenarios"]
    )
    assert report["interpretation"]["apf_contribution_claim"] == "not_claimed"
    assert report["boundary_notes"] == [
        "headless_metrics_are_source_of_truth",
        "videos_are_illustrative_when_present",
        "deterministic_tuning_does_not_claim_training_convergence",
        "simulator_evidence_is_not_real_flight_safety_evidence",
    ]
    assert "NaN" not in raw_report
    assert "Infinity" not in raw_report

    manifest_path = Path(report["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "multi_scenario_evaluation_report"
    assert manifest["lineage"]["evaluation_level"] == "deterministic_multi_scenario_tuning"
    assert {reference["role"] for reference in manifest["outputs"]} == {"multi_scenario_report"}


def test_open_corridor_scenario_uses_effective_zero_safety_threshold(tmp_path: Path):
    output = tmp_path / "open_corridor.json"
    config = MultiScenarioEvaluationConfig(
        scenarios=(ScenarioCase(identifier="open", scenario=Stage1Scenario(obstacles=(), max_steps=30)),),
        output=output,
    )

    report = run_multi_scenario_evaluation(config)

    scenario = report["scenarios"][0]
    assert scenario["metrics"]["success"] is True
    assert scenario["acceptance"]["minimum_safety_distance"] == 0.0
    assert scenario["acceptance"]["accepted"] is True
    assert report["readiness"]["all_scenarios_have_accepted_candidate"] is True


def test_failure_breakdown_counts_unsafe_distance_when_threshold_blocks_acceptance(tmp_path: Path):
    output = tmp_path / "unsafe.json"
    config = MultiScenarioEvaluationConfig(
        scenarios=(
            ScenarioCase(
                identifier="strict-safety",
                scenario=Stage1Scenario(),
                minimum_safety_distance=99.0,
            ),
        ),
        output=output,
    )

    report = run_multi_scenario_evaluation(config)

    scenario = report["scenarios"][0]
    assert scenario["acceptance"]["accepted"] is False
    assert scenario["failure_breakdown"]["unsafe_distance"] == 1


def test_multi_scenario_evaluation_rejects_duplicate_scenario_ids(tmp_path: Path):
    with pytest.raises(ValueError, match="unique"):
        MultiScenarioEvaluationConfig(
            scenarios=(
                ScenarioCase(identifier="duplicate", scenario=Stage1Scenario()),
                ScenarioCase(identifier="duplicate", scenario=Stage1Scenario(obstacles=())),
            ),
            output=tmp_path / "out.json",
        )


def test_scenario_case_rejects_blank_identifier():
    with pytest.raises(ValueError, match="identifier"):
        ScenarioCase(identifier=" ", scenario=Stage1Scenario(obstacles=(ObstacleState(position=(1, 0, 0), radius=0.2),)))
