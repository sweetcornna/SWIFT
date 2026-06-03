import json
from pathlib import Path

import pytest

from swift.experiments.convergence_gate import (
    ConvergenceGateConfig,
    ConvergenceThresholds,
    run_convergence_gate,
)


def test_convergence_gate_accepts_long_training_report_and_writes_manifest(tmp_path: Path):
    input_path = tmp_path / "ablation.json"
    output_path = tmp_path / "convergence_gate.json"
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

    raw_report = output_path.read_text(encoding="utf-8")
    assert json.loads(raw_report) == report
    assert report["record_type"] == "training_convergence_gate_report"
    assert report["source_record_type"] == "training_ablation_report"
    assert report["best_variant"] == "ppo_hca_apf"
    assert report["readiness"]["convergence_claim"] is True
    assert report["readiness"]["evidence_level"] == "long_training_convergence"
    assert {gate["name"] for gate in report["gates"]} >= {"required_variants_completed"}
    assert all(gate["passed"] for gate in report["gates"])
    assert report["boundary_notes"] == [
        "convergence_claim_requires_long_training_evidence",
        "gate_uses_recorded_metrics_not_video_or_manual_observation",
        "simulator_convergence_is_not_real_flight_safety_evidence",
    ]
    assert "NaN" not in raw_report
    assert "Infinity" not in raw_report

    manifest_path = Path(report["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "training_convergence_gate_report"
    assert {reference["role"] for reference in manifest["inputs"]} == {"source_training_report"}
    assert {reference["role"] for reference in manifest["outputs"]} == {"convergence_gate_report"}


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
