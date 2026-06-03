from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def write_stage1_report(
    ppo_summary_path: str | Path,
    tuning_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    ppo_path = Path(ppo_summary_path)
    tuning_path = Path(tuning_summary_path)
    report_path = Path(output_path)
    ppo_summary = _read_json_object(ppo_path)
    tuning_summary = _read_json_object(tuning_path)
    report = build_stage1_report(
        ppo_summary=ppo_summary,
        tuning_summary=tuning_summary,
        ppo_summary_path=ppo_path,
        tuning_summary_path=tuning_path,
        output_path=report_path,
    )
    serialized = json.dumps(report, allow_nan=False, sort_keys=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"{serialized}\n", encoding="utf-8")
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_stage1_report(
    *,
    ppo_summary: dict[str, Any],
    tuning_summary: dict[str, Any],
    ppo_summary_path: Path,
    tuning_summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _validate_ppo_summary(ppo_summary)
    ppo_path = Path(ppo_summary_path)
    tuning_path = Path(tuning_summary_path)
    report_path = Path(output_path)
    training = ppo_summary.get("training", {})
    metrics = ppo_summary.get("metrics", {})
    artifacts = ppo_summary.get("artifacts", {})
    if not isinstance(training, dict) or not isinstance(metrics, dict) or not isinstance(artifacts, dict):
        raise ValueError("ppo summary must contain object training, metrics, and artifacts sections")

    baseline = _mapping(tuning_summary.get("baseline"), "tuning baseline")
    best_candidate = tuning_summary.get("best_candidate")
    if best_candidate is not None:
        best_candidate = _mapping(best_candidate, "best_candidate")
    acceptance = _mapping(tuning_summary.get("acceptance"), "tuning acceptance")
    best_metrics = _mapping(best_candidate.get("metrics"), "best_candidate metrics") if best_candidate else {}

    readiness = {
        "training_smoke_complete": int(training.get("updates", 0)) > 0
        and int(training.get("total_timesteps", 0)) > 0,
        "tuning_accepted": bool(acceptance.get("accepted", False)),
        "baseline_collided": bool(baseline.get("collided", False)),
        "best_candidate_success": bool(best_metrics.get("success", False)),
        "collision_to_success": bool(baseline.get("collided", False))
        and bool(best_metrics.get("success", False))
        and not bool(best_metrics.get("collided", False)),
    }

    return {
        "schema_version": 1,
        "record_type": "stage1_training_tuning_report",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": str(ppo_summary.get("stage", "stage1")),
        "variant": str(ppo_summary.get("variant", "ppo_mlp")),
        "training": {
            "run_id": str(ppo_summary.get("run_id", "")),
            "total_timesteps": int(training.get("total_timesteps", 0)),
            "updates": int(training.get("updates", 0)),
            "episodes_completed": int(training.get("episodes_completed", 0)),
            "metrics": dict(metrics),
            "artifacts": dict(artifacts),
        },
        "tuning": {
            "accepted": bool(acceptance.get("accepted", False)),
            "acceptance": dict(acceptance),
            "baseline": dict(baseline),
            "best_candidate": dict(best_candidate) if best_candidate else None,
        },
        "readiness": readiness,
        "evidence": {
            "ppo_summary_json": str(ppo_path),
            "tuning_summary_json": str(tuning_path),
            "report_json": str(report_path),
            "ppo_artifact_summary_json": str(artifacts.get("summary_json", ppo_path)),
            "training_history_jsonl": str(artifacts.get("training_history_jsonl", "")),
            "checkpoint_path": str(artifacts.get("checkpoint_path", "")),
        },
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _validate_ppo_summary(summary: dict[str, Any]) -> None:
    if summary.get("record_type") != "ppo_training_smoke":
        raise ValueError("ppo summary record_type must be ppo_training_smoke")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value
