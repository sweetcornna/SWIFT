import json
from pathlib import Path

import pytest

from swift.experiments.stage1_report import write_stage1_report


def test_stage1_report_merges_training_and_tuning_evidence(tmp_path: Path):
    ppo_summary_path = tmp_path / "ppo.json"
    tuning_summary_path = tmp_path / "tuning.json"
    output_path = tmp_path / "report.json"
    ppo_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ppo_training_smoke",
                "stage": "stage1",
                "variant": "ppo_mlp",
                "run_id": "stage1_ppo_seed-3",
                "training": {
                    "total_timesteps": 128,
                    "updates": 2,
                    "episodes_completed": 12,
                },
                "metrics": {
                    "success_rate": 0.25,
                    "collision_rate": 0.5,
                    "timeout_rate": 0.25,
                    "average_episode_return": -3.5,
                },
                "artifacts": {
                    "summary_json": "outputs/training/ppo.json",
                    "training_history_jsonl": "outputs/episodes/stage1/ppo.jsonl",
                    "checkpoint_path": "checkpoints/stage1/ppo.ckpt",
                },
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    tuning_summary_path.write_text(
        json.dumps(
            {
                "baseline": {
                    "success": False,
                    "collided": True,
                    "timed_out": False,
                },
                "best_candidate": {
                    "accepted": True,
                    "metrics": {
                        "success": True,
                        "collided": False,
                        "timed_out": False,
                        "minimum_safety_distance": 0.44,
                    },
                },
                "acceptance": {
                    "accepted": True,
                    "candidate_count": 4,
                    "accepted_candidate_count": 1,
                    "minimum_safety_distance": 0.3,
                },
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = write_stage1_report(ppo_summary_path, tuning_summary_path, output_path)

    raw_report = output_path.read_text(encoding="utf-8")
    assert json.loads(raw_report) == report
    assert report["schema_version"] == 1
    assert report["record_type"] == "stage1_training_tuning_report"
    assert report["stage"] == "stage1"
    assert report["training"]["run_id"] == "stage1_ppo_seed-3"
    assert report["training"]["updates"] == 2
    assert report["training"]["metrics"]["collision_rate"] == 0.5
    assert report["tuning"]["accepted"] is True
    assert report["readiness"] == {
        "training_smoke_complete": True,
        "tuning_accepted": True,
        "baseline_collided": True,
        "best_candidate_success": True,
        "collision_to_success": True,
    }
    assert report["evidence"]["ppo_summary_json"] == str(ppo_summary_path)
    assert report["evidence"]["tuning_summary_json"] == str(tuning_summary_path)
    assert report["evidence"]["report_json"] == str(output_path)
    manifest_path = Path(report["evidence"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["subject_record_type"] == "stage1_training_tuning_report"
    assert {reference["role"] for reference in manifest["inputs"]} == {"ppo_summary_json", "tuning_summary_json"}
    assert {reference["role"] for reference in manifest["outputs"]} == {"report_json"}
    assert "NaN" not in raw_report
    assert "Infinity" not in raw_report


def test_stage1_report_rejects_wrong_training_record_type(tmp_path: Path):
    ppo_summary_path = tmp_path / "ppo.json"
    tuning_summary_path = tmp_path / "tuning.json"
    output_path = tmp_path / "report.json"
    ppo_summary_path.write_text('{"record_type": "other"}', encoding="utf-8")
    tuning_summary_path.write_text('{"acceptance": {"accepted": true}}', encoding="utf-8")

    with pytest.raises(ValueError, match="ppo_training_smoke"):
        write_stage1_report(ppo_summary_path, tuning_summary_path, output_path)
