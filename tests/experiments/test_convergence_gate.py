import json
from pathlib import Path

import pytest

from swift.experiments.convergence_gate import (
    ConvergenceGateConfig,
    ConvergenceThresholds,
    run_convergence_gate,
)


def test_convergence_gate_accepts_holdout_checkpoint_report_and_writes_manifest(tmp_path: Path):
    input_path = tmp_path / "holdout.json"
    output_path = tmp_path / "convergence_gate.json"
    input_path.write_text(json.dumps(_holdout_report(seed_count=2)), encoding="utf-8")

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_seed_count=2,
                min_total_timesteps=4096,
                min_episodes_completed=10,
                required_evidence_level="long_training_convergence",
            ),
        )
    )

    raw_report = output_path.read_text(encoding="utf-8")
    assert json.loads(raw_report) == report
    assert report["record_type"] == "training_convergence_gate_report"
    assert report["source_record_type"] == "multi_seed_checkpoint_holdout_report"
    assert report["best_variant"] == "ppo_hca_apf"
    assert report["readiness"]["convergence_claim"] is True
    assert report["readiness"]["evidence_level"] == "long_training_convergence"
    assert {gate["name"] for gate in report["gates"]} >= {"holdout_checkpoint_evidence", "required_variants_completed"}
    assert all(gate["passed"] for gate in report["gates"])
    assert report["boundary_notes"] == [
        "convergence_claim_requires_long_training_evidence",
        "convergence_claim_requires_holdout_checkpoint_evidence",
        "gate_uses_recorded_metrics_not_video_or_manual_observation",
        "simulator_convergence_is_not_real_flight_safety_evidence",
    ]
    assert "NaN" not in raw_report
    assert "Infinity" not in raw_report

    manifest_path = Path(report["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "training_convergence_gate_report"
    assert {reference["role"] for reference in manifest["inputs"]} == {"source_convergence_evidence_report"}
    assert {reference["role"] for reference in manifest["outputs"]} == {"convergence_gate_report"}


def test_convergence_gate_rejects_raw_long_training_ablation_without_holdout_evidence(tmp_path: Path):
    input_path = tmp_path / "ablation.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(json.dumps(_ablation_report(evidence_level="long_training_convergence")), encoding="utf-8")

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_total_timesteps=4096,
                min_episodes_completed=10,
                required_evidence_level="long_training_convergence",
            ),
        )
    )

    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["readiness"]["convergence_claim"] is False
    assert failed["holdout_checkpoint_evidence"]["actual"] == "training_ablation_report"
    assert failed["holdout_checkpoint_evidence"]["expected"] == "multi_seed_checkpoint_holdout_report"


def test_convergence_gate_rejects_cpu_smoke_even_when_metrics_pass(tmp_path: Path):
    input_path = tmp_path / "smoke_ablation.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(json.dumps(_ablation_report(evidence_level="cpu_smoke_ablation")), encoding="utf-8")

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(required_evidence_level="long_training_convergence"),
        )
    )

    assert report["readiness"]["convergence_claim"] is False
    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert failed["evidence_level"]["actual"] == "cpu_smoke_ablation"
    assert failed["evidence_level"]["expected"] == "long_training_convergence"


def test_convergence_gate_rejects_missing_required_ablation_variant(tmp_path: Path):
    input_path = tmp_path / "missing_variant.json"
    output_path = tmp_path / "gate.json"
    report = _ablation_report(evidence_level="long_training_convergence")
    report["variants"] = [
        variant for variant in report["variants"] if variant["variant"] != "ppo_hca"
    ]
    input_path.write_text(json.dumps(report), encoding="utf-8")

    gated = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_total_timesteps=4096,
                min_episodes_completed=10,
                required_evidence_level="long_training_convergence",
            ),
        )
    )

    failed = {gate["name"]: gate for gate in gated["gates"] if not gate["passed"]}
    assert gated["readiness"]["convergence_claim"] is False
    assert failed["required_variants_completed"]["actual"] == ["ppo_hca_apf", "ppo_mlp"]
    assert failed["required_variants_completed"]["expected"] == ["ppo_mlp", "ppo_hca", "ppo_hca_apf"]


def test_convergence_gate_rejects_unknown_best_variant_instead_of_falling_back(tmp_path: Path):
    input_path = tmp_path / "unknown_best.json"
    report = _ablation_report(evidence_level="long_training_convergence")
    report["best_variant"] = "missing_variant"
    input_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="best_variant must match one variant"):
        run_convergence_gate(ConvergenceGateConfig(input_report=input_path))


def test_convergence_gate_rejects_out_of_range_rate_metrics(tmp_path: Path):
    input_path = tmp_path / "bad_rate.json"
    report = _ablation_report(evidence_level="long_training_convergence")
    report["variants"][2]["metrics"]["success_rate"] = 1.2
    input_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="success_rate must be between 0 and 1"):
        run_convergence_gate(ConvergenceGateConfig(input_report=input_path))


def test_convergence_gate_requires_explicit_completed_true_for_required_variants(tmp_path: Path):
    input_path = tmp_path / "implicit_completed.json"
    output_path = tmp_path / "gate.json"
    report = _ablation_report(evidence_level="long_training_convergence")
    for variant in report["variants"]:
        del variant["completed"]
    input_path.write_text(json.dumps(report), encoding="utf-8")

    gated = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_total_timesteps=4096,
                min_episodes_completed=10,
            ),
        )
    )

    failed = {gate["name"]: gate for gate in gated["gates"] if not gate["passed"]}
    assert gated["readiness"]["convergence_claim"] is False
    assert failed["required_variants_completed"]["actual"] == []


def test_convergence_gate_rejects_mismatched_evidence_levels(tmp_path: Path):
    input_path = tmp_path / "mismatched_evidence.json"
    output_path = tmp_path / "gate.json"
    report = _ablation_report(evidence_level="long_training_convergence")
    report["lineage"]["evidence_level"] = "cpu_smoke_ablation"
    input_path.write_text(json.dumps(report), encoding="utf-8")

    gated = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_total_timesteps=4096,
                min_episodes_completed=10,
            ),
        )
    )

    failed = {gate["name"]: gate for gate in gated["gates"] if not gate["passed"]}
    assert gated["readiness"]["convergence_claim"] is False
    assert failed["evidence_level_consistency"]["actual"] == [
        "cpu_smoke_ablation",
        "long_training_convergence",
    ]


def test_convergence_gate_audits_multi_scenario_report_schema(tmp_path: Path):
    input_path = tmp_path / "multi_scenario.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(json.dumps(_multi_scenario_report()), encoding="utf-8")

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                convergence_claim_allowed=False,
                min_success_rate=0.95,
                max_collision_rate=0.0,
                max_timeout_rate=0.05,
                min_total_timesteps=1,
                min_episodes_completed=1,
                required_evidence_level="deterministic_multi_scenario_tuning",
                required_variants=("multi_scenario_policy_tuning",),
            ),
        )
    )

    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["source_record_type"] == "multi_scenario_evaluation_report"
    assert report["best_variant"] == "multi_scenario_policy_tuning"
    assert report["readiness"]["convergence_claim"] is False
    assert failed["convergence_claim_allowed"]["actual"] is False
    assert "required_variants_completed" not in failed


def test_convergence_gate_rejects_raw_multi_seed_report_without_holdout_evidence(tmp_path: Path):
    input_path = tmp_path / "multi_seed.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(json.dumps(_multi_seed_report(seed_count=2)), encoding="utf-8")

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_seed_count=2,
                min_total_timesteps=4096,
                min_episodes_completed=10,
            ),
        )
    )

    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["readiness"]["convergence_claim"] is False
    assert failed["holdout_checkpoint_evidence"]["actual"] == "multi_seed_training_report"
    assert failed["holdout_checkpoint_evidence"]["expected"] == "multi_seed_checkpoint_holdout_report"


def test_convergence_gate_rejects_multi_seed_report_with_too_few_seeds(tmp_path: Path):
    input_path = tmp_path / "multi_seed.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(json.dumps(_multi_seed_report(seed_count=1)), encoding="utf-8")

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_seed_count=2,
                min_total_timesteps=4096,
                min_episodes_completed=10,
            ),
        )
    )

    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["readiness"]["convergence_claim"] is False
    assert failed["seed_count"]["actual"] == 1
    assert failed["seed_count"]["expected"] == ">=2"


def test_convergence_gate_uses_minimum_seed_timesteps_for_multi_seed_report(tmp_path: Path):
    input_path = tmp_path / "multi_seed_short_per_seed.json"
    output_path = tmp_path / "gate.json"
    report = _multi_seed_report(seed_count=3)
    for variant in report["variants"]:
        variant["training"]["total_timesteps"] = 100002
        variant["seed_metrics"] = [
            {
                "seed": 0,
                "training": {"total_timesteps": 33334, "updates": 8, "episodes_completed": 10},
                "metrics": variant["metrics"],
            },
            {
                "seed": 1,
                "training": {"total_timesteps": 33334, "updates": 8, "episodes_completed": 10},
                "metrics": variant["metrics"],
            },
            {
                "seed": 2,
                "training": {"total_timesteps": 33334, "updates": 8, "episodes_completed": 10},
                "metrics": variant["metrics"],
            },
        ]
    input_path.write_text(json.dumps(report), encoding="utf-8")

    gated = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=input_path,
            output=output_path,
            thresholds=ConvergenceThresholds(
                min_success_rate=0.9,
                max_collision_rate=0.0,
                max_timeout_rate=0.1,
                min_seed_count=3,
                min_total_timesteps=100000,
                min_episodes_completed=10,
            ),
        )
    )

    failed = {gate["name"]: gate for gate in gated["gates"] if not gate["passed"]}
    assert gated["readiness"]["convergence_claim"] is False
    assert failed["total_timesteps"]["actual"] == 33334
    assert failed["total_timesteps"]["expected"] == ">=100000"


def test_convergence_gate_rejects_non_finite_metrics(tmp_path: Path):
    input_path = tmp_path / "bad.json"
    bad_report = _ablation_report(evidence_level="long_training_convergence")
    bad_report["variants"][0]["metrics"]["success_rate"] = float("nan")
    input_path.write_text(json.dumps(bad_report), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        run_convergence_gate(ConvergenceGateConfig(input_report=input_path))


def _ablation_report(*, evidence_level: str) -> dict[str, object]:
    metrics = {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "average_episode_return": 42.0,
    }
    return {
        "schema_version": 1,
        "record_type": "training_ablation_report",
        "stage": "stage4",
        "variant": "ablation",
        "run_id": "stage4_ablation_seed-7_cfg-test_20260603t000000z",
        "lineage": {
            "evidence_level": evidence_level,
        },
        "best_variant": "ppo_hca_apf",
        "variants": [
            _variant("ppo_mlp", metrics),
            _variant("ppo_hca", metrics),
            _variant("ppo_hca_apf", metrics),
        ],
        "readiness": {
            "all_variants_completed": True,
            "convergence_claim": False,
            "evidence_level": evidence_level,
        },
    }


def _variant(name: str, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "variant": name,
        "score": 100.42,
        "completed": True,
        "metrics": metrics,
        "training": {
            "total_timesteps": 8192,
            "updates": 16,
            "episodes_completed": 24,
        },
    }


def _holdout_report(*, seed_count: int) -> dict[str, object]:
    report = _multi_seed_report(seed_count=seed_count)
    report["record_type"] = "multi_seed_checkpoint_holdout_report"
    report["variant"] = "multi_seed_checkpoint_holdout"
    report["source_record_type"] = "multi_seed_training_report"
    report["source_run_id"] = "stage4_multi-seed-training_seed-0_cfg-test_20260603t000000z"
    report["training_seeds"] = list(range(seed_count))
    report["holdout_seeds"] = [10000, 11000]
    report["holdout_seed_count"] = 2
    report["readiness"] = {
        "all_checkpoints_evaluated": True,
        "training_seed_count": seed_count,
        "holdout_seed_count": 2,
        "convergence_claim": False,
        "evidence_level": "long_training_convergence",
    }
    for variant in report["variants"]:
        variant["training_seed_count"] = seed_count
        variant["holdout_seed_count"] = 2
        variant["holdout_checkpoint_count"] = seed_count * 2
    return report


def _multi_scenario_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "multi_scenario_evaluation_report",
        "stage": "stage4",
        "variant": "multi_scenario_policy_tuning",
        "run_id": "stage4_multi-scenario-policy-tuning_seed-0_cfg-test_20260603t000000z",
        "lineage": {"evidence_level": "deterministic_multi_scenario_tuning"},
        "summary": {
            "scenario_count": 2,
            "globally_accepted_candidate_count": 1,
        },
        "ranking": [
            {
                "rank": 1,
                "accepted_all": True,
                "metrics": {
                    "success_rate": 1.0,
                    "collision_rate": 0.0,
                    "timeout_rate": 0.0,
                    "average_path_length": 8.0,
                    "average_path_smoothness": 0.2,
                    "minimum_safety_distance": 0.4,
                },
                "scenarios": [],
            }
        ],
        "scenarios": [
            {"scenario_id": "fixed", "metrics": {"steps": 12}},
            {"scenario_id": "open", "metrics": {"steps": 8}},
        ],
        "readiness": {
            "all_scenarios_evaluated": True,
            "global_candidate_accepted": True,
            "convergence_claim": False,
            "evidence_level": "deterministic_multi_scenario_tuning",
        },
    }


def _multi_seed_report(*, seed_count: int) -> dict[str, object]:
    metrics = {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "average_episode_return": 42.0,
    }
    return {
        "schema_version": 1,
        "record_type": "multi_seed_training_report",
        "stage": "stage4",
        "variant": "multi_seed_training",
        "run_id": "stage4_multi-seed-training_seed-0_cfg-test_20260603t000000z",
        "lineage": {"evidence_level": "long_training_convergence"},
        "best_variant": "ppo_hca_apf",
        "seed_count": seed_count,
        "seeds": list(range(seed_count)),
        "variants": [
            _multi_seed_variant("ppo_mlp", metrics, seed_count),
            _multi_seed_variant("ppo_hca", metrics, seed_count),
            _multi_seed_variant("ppo_hca_apf", metrics, seed_count),
        ],
        "readiness": {
            "all_seeds_completed": True,
            "convergence_claim": False,
            "evidence_level": "long_training_convergence",
        },
    }


def _multi_seed_variant(
    name: str,
    metrics: dict[str, float],
    seed_count: int,
) -> dict[str, object]:
    return {
        "variant": name,
        "completed": True,
        "seed_count": seed_count,
        "completed_seeds": seed_count,
        "metrics": metrics,
        "training": {
            "total_timesteps": 8192,
            "updates": 16,
            "episodes_completed": 24,
        },
    }
