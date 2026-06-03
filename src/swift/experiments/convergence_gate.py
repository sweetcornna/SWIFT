from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import json
import math

from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)


BOUNDARY_NOTES = (
    "convergence_claim_requires_long_training_evidence",
    "gate_uses_recorded_metrics_not_video_or_manual_observation",
    "simulator_convergence_is_not_real_flight_safety_evidence",
)


@dataclass(frozen=True)
class ConvergenceThresholds:
    convergence_claim_allowed: bool = True
    min_success_rate: float = 0.95
    max_collision_rate: float = 0.0
    max_timeout_rate: float = 0.05
    min_total_timesteps: int = 4096
    min_episodes_completed: int = 10
    required_evidence_level: str = "long_training_convergence"
    required_variants: tuple[str, ...] = ("ppo_mlp", "ppo_hca", "ppo_hca_apf")

    def __post_init__(self) -> None:
        _unit_interval("min_success_rate", self.min_success_rate)
        _unit_interval("max_collision_rate", self.max_collision_rate)
        _unit_interval("max_timeout_rate", self.max_timeout_rate)
        if self.min_total_timesteps <= 0:
            raise ValueError("min_total_timesteps must be positive")
        if self.min_episodes_completed <= 0:
            raise ValueError("min_episodes_completed must be positive")
        object.__setattr__(self, "min_total_timesteps", int(self.min_total_timesteps))
        object.__setattr__(self, "min_episodes_completed", int(self.min_episodes_completed))
        required_evidence_level = str(self.required_evidence_level).strip() or "long_training_convergence"
        object.__setattr__(
            self,
            "required_evidence_level",
            required_evidence_level,
        )
        object.__setattr__(
            self,
            "convergence_claim_allowed",
            bool(self.convergence_claim_allowed) and required_evidence_level == "long_training_convergence",
        )
        variants = tuple(str(variant).strip() for variant in self.required_variants if str(variant).strip())
        if not variants:
            raise ValueError("required_variants must contain at least one variant")
        object.__setattr__(self, "required_variants", variants)


@dataclass(frozen=True)
class ConvergenceGateConfig:
    input_report: Path
    output: Path | None = None
    thresholds: ConvergenceThresholds = field(default_factory=ConvergenceThresholds)
    stage: str = "stage4"
    variant: str = "convergence_gate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_report", Path(self.input_report))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "stage", str(self.stage).strip() or "stage4")
        object.__setattr__(self, "variant", str(self.variant).strip() or "convergence_gate")


def run_convergence_gate(config: ConvergenceGateConfig) -> dict[str, Any]:
    source = _load_source_report(config.input_report)
    candidate = _best_candidate(source)
    gates = _gates(source, candidate, config.thresholds)
    convergence_claim = all(bool(gate["passed"]) for gate in gates)
    run_id = build_run_id(
        stage=config.stage,
        variant=config.variant,
        seed=0,
        config_hash=_config_hash(source, config.thresholds),
        started_at_utc=datetime.now(UTC),
    )
    writer = ExperimentArtifactWriter()
    paths = writer.paths_for(config.stage, config.variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json
    evidence_level = _evidence_level(source)
    report = {
        "schema_version": 1,
        "record_type": "training_convergence_gate_report",
        "stage": config.stage,
        "variant": config.variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_record_type": str(source.get("record_type", "")),
        "source_run_id": str(source.get("run_id", "")),
        "best_variant": str(candidate["variant"]),
        "thresholds": asdict(config.thresholds),
        "metrics": dict(candidate["metrics"]),
        "training": dict(candidate["training"]),
        "gates": gates,
        "readiness": {
            "convergence_claim": convergence_claim,
            "evidence_level": evidence_level,
            "passed_gates": sum(1 for gate in gates if gate["passed"]),
            "required_gates": len(gates),
        },
        "boundary_notes": list(BOUNDARY_NOTES),
        "artifacts": {
            "summary_json": str(output_path),
            "manifest_json": str(manifest_path),
        },
    }
    _assert_json_safe(report)
    writer.write_summary(output_path, report)
    writer.write_manifest(
        manifest_path,
        build_artifact_manifest(
            subject_record_type=report["record_type"],
            run_id=run_id,
            stage=config.stage,
            variant=config.variant,
            lineage={
                "source_record_type": report["source_record_type"],
                "source_run_id": report["source_run_id"],
                "evidence_level": evidence_level,
                "convergence_claim": convergence_claim,
            },
            inputs=[artifact_reference(config.input_report, role="source_training_report")],
            outputs=[artifact_reference(output_path, role="convergence_gate_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _load_source_report(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("input report must be a JSON object")
    _assert_json_safe(source)
    return source


def _best_candidate(source: dict[str, Any]) -> dict[str, Any]:
    variants = source.get("variants")
    best_variant = source.get("best_variant")
    if isinstance(variants, list) and best_variant is not None:
        for variant in variants:
            if isinstance(variant, dict) and variant.get("variant") == best_variant:
                return _candidate_record(variant)
        raise ValueError("best_variant must match one variant")
    if isinstance(variants, list) and variants:
        return _candidate_record(variants[0])
    if "metrics" in source and "training" in source:
        return _candidate_record(
            {
                "variant": source.get("variant", "unknown"),
                "metrics": source["metrics"],
                "training": source["training"],
            }
        )
    raise ValueError("input report must contain variants or metrics/training")


def _candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(candidate.get("metrics", {}))
    training = dict(candidate.get("training", {}))
    for key in ("success_rate", "collision_rate", "timeout_rate", "average_episode_return"):
        _finite_metric(metrics, key)
    for key in ("total_timesteps", "updates", "episodes_completed"):
        if int(training.get(key, 0)) < 0:
            raise ValueError(f"{key} must be non-negative")
    return {
        "variant": str(candidate.get("variant", "unknown")),
        "metrics": metrics,
        "training": training,
    }


def _gates(
    source: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: ConvergenceThresholds,
) -> list[dict[str, Any]]:
    metrics = candidate["metrics"]
    training = candidate["training"]
    evidence_level = _evidence_level(source)
    evidence_levels = _evidence_level_values(source)
    completed_variants = _completed_variants(source)
    return [
        _gate(
            "convergence_claim_allowed",
            thresholds.convergence_claim_allowed,
            True,
            thresholds.convergence_claim_allowed,
        ),
        _gate(
            "evidence_level_consistency",
            evidence_levels,
            "single evidence level",
            len(evidence_levels) <= 1,
        ),
        _gate("evidence_level", evidence_level, thresholds.required_evidence_level, evidence_level == thresholds.required_evidence_level),
        _gate(
            "required_variants_completed",
            completed_variants,
            list(thresholds.required_variants),
            all(variant in completed_variants for variant in thresholds.required_variants),
        ),
        _gate(
            "success_rate",
            _finite_metric(metrics, "success_rate"),
            f">={thresholds.min_success_rate}",
            _finite_metric(metrics, "success_rate") >= thresholds.min_success_rate,
        ),
        _gate(
            "collision_rate",
            _finite_metric(metrics, "collision_rate"),
            f"<={thresholds.max_collision_rate}",
            _finite_metric(metrics, "collision_rate") <= thresholds.max_collision_rate,
        ),
        _gate(
            "timeout_rate",
            _finite_metric(metrics, "timeout_rate"),
            f"<={thresholds.max_timeout_rate}",
            _finite_metric(metrics, "timeout_rate") <= thresholds.max_timeout_rate,
        ),
        _gate(
            "total_timesteps",
            int(training.get("total_timesteps", 0)),
            f">={thresholds.min_total_timesteps}",
            int(training.get("total_timesteps", 0)) >= thresholds.min_total_timesteps,
        ),
        _gate(
            "episodes_completed",
            int(training.get("episodes_completed", 0)),
            f">={thresholds.min_episodes_completed}",
            int(training.get("episodes_completed", 0)) >= thresholds.min_episodes_completed,
        ),
    ]


def _completed_variants(source: dict[str, Any]) -> list[str]:
    variants = source.get("variants", [])
    if not isinstance(variants, list):
        return []
    completed = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        if variant.get("completed") is True:
            completed.append(str(variant.get("variant", "")))
    return sorted(name for name in completed if name)


def _gate(name: str, actual: Any, expected: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "passed": bool(passed),
    }


def _evidence_level(source: dict[str, Any]) -> str:
    readiness = source.get("readiness", {})
    if isinstance(readiness, dict) and readiness.get("evidence_level"):
        return str(readiness["evidence_level"])
    lineage = source.get("lineage", {})
    if isinstance(lineage, dict) and lineage.get("evidence_level"):
        return str(lineage["evidence_level"])
    return "unknown"


def _evidence_level_values(source: dict[str, Any]) -> list[str]:
    values = []
    readiness = source.get("readiness", {})
    if isinstance(readiness, dict) and readiness.get("evidence_level"):
        values.append(str(readiness["evidence_level"]))
    lineage = source.get("lineage", {})
    if isinstance(lineage, dict) and lineage.get("evidence_level"):
        values.append(str(lineage["evidence_level"]))
    return sorted(set(values))


def _finite_metric(metrics: dict[str, Any], key: str) -> float:
    value = float(metrics.get(key, 0.0))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if key in {"success_rate", "collision_rate", "timeout_rate"} and not (0.0 <= value <= 1.0):
        raise ValueError(f"{key} must be between 0 and 1")
    return value


def _unit_interval(name: str, value: float) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _config_hash(source: dict[str, Any], thresholds: ConvergenceThresholds) -> str:
    import hashlib

    material = json.dumps(
        {
            "source_run_id": source.get("run_id", ""),
            "thresholds": asdict(thresholds),
        },
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _assert_json_safe(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("report contains a non-finite float")
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe(item)
