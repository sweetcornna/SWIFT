import json
from pathlib import Path

import pytest

from swift.hover.publication import HoverPublicationError, compact_evaluation, publish_120k_runs


def test_compact_evaluation_omits_cases_and_retains_evidence():
    report = {
        "schema_version": 4, "evaluation_id": "eval", "model_kind": "robust_best_model",
        "model_sha256_before": "a", "normalization": {"stats_sha256_before": "b"},
        "evaluation_seed": 101, "case_count": 100, "environment": {"action": "CT_ATT_YAWRATE_V1"},
        "sampling": {}, "safety_definition": "safe", "aggregates": {"safety_completion_count": 100},
        "acceptance_thresholds": {}, "acceptance_checks": {"accepted": True},
        "harness_validity_checks": {}, "evaluation_valid": True, "verdict": "accepted", "cases": [{"large": "bulk"}],
    }
    compact = compact_evaluation(report)
    assert "cases" not in compact
    assert compact["aggregates"]["safety_completion_count"] == 100
    assert compact["record_type"] == "swift_compact_robust_hover_evaluation"


def test_compact_evaluation_requires_verdict():
    with pytest.raises(HoverPublicationError, match="missing"):
        compact_evaluation({})


def test_publication_rejects_accepted_report_without_full_100_case_contract(tmp_path, monkeypatch):
    training_runs = []
    evaluation_dirs = []
    for seed in (1, 2, 3):
        training = tmp_path / f"training-{seed}"
        evaluation = tmp_path / f"evaluation-{seed}"
        training.mkdir()
        evaluation.mkdir()
        (training / "summary.json").write_text("{}", encoding="utf-8")
        (evaluation / "evaluation.json").write_text(json.dumps({
            "model_kind": "robust_best_model",
            "verdict": "accepted",
            "evaluation_valid": True,
            "case_count": 2,
            "harness_validity_checks": {"full_sampling_contract": False},
            "normalization": {},
        }), encoding="utf-8")
        training_runs.append(training)
        evaluation_dirs.append(evaluation)

    monkeypatch.setattr("swift.hover.publication.load_bound_training_summary", lambda path: {
        "training": {"train_seed": int(Path(path).name.rsplit("-", 1)[1]), "requested_timesteps": 120_000},
        "model_normalization_bindings": {"robust_best_model": {"model_sha256": "model", "stats_sha256": "stats"}},
    })
    with pytest.raises(HoverPublicationError, match="100-case"):
        publish_120k_runs(training_runs, evaluation_dirs, tmp_path / "published")
