from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import math

from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)
from swift.experiments.pybullet_evidence import (
    FINAL_HOLDOUT_SEED_END,
    FINAL_HOLDOUT_SEED_START,
)


BOUNDARY_NOTES = (
    "gate_uses_randomized_pybullet_holdout_metrics",
    "holdout_layouts_must_not_be_used_for_hyperparameter_tuning",
    "simulator_robustness_is_not_real_flight_safety_evidence",
)


@dataclass(frozen=True)
class PyBulletRobustnessThresholds:
    min_training_seed_count: int = 3
    min_total_timesteps: int = 100_000
    min_holdout_episodes_per_checkpoint: int = 100
    min_success_rate: float = 0.95
    max_collision_rate: float = 0.0
    max_timeout_rate: float = 0.05
    min_average_minimum_safety_distance: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "min_training_seed_count",
            "min_total_timesteps",
            "min_holdout_episodes_per_checkpoint",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("min_success_rate", "max_collision_rate", "max_timeout_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        safety = float(self.min_average_minimum_safety_distance)
        if not math.isfinite(safety) or safety < 0.0:
            raise ValueError("min_average_minimum_safety_distance must be non-negative")
        object.__setattr__(self, "min_average_minimum_safety_distance", safety)


@dataclass(frozen=True)
class PyBulletRobustnessGateConfig:
    input_report: Path
    output: Path | None = None
    thresholds: PyBulletRobustnessThresholds = field(default_factory=PyBulletRobustnessThresholds)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_report", Path(self.input_report))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))


def run_pybullet_robustness_gate(config: PyBulletRobustnessGateConfig) -> dict[str, Any]:
    source = _load_source(config.input_report)
    gates = _gates(source, config.thresholds)
    robustness_claim = all(bool(gate["passed"]) for gate in gates)
    stage = str(source.get("stage", "stage1"))
    variant = "pybullet_robustness_gate"
    run_id = build_run_id(
        stage=stage,
        variant=variant,
        seed=int(source.get("holdout", {}).get("seed_start", 0)),
        config_hash=_config_hash(source, config.thresholds),
        started_at_utc=datetime.now(UTC),
    )
    writer = ExperimentArtifactWriter()
    paths = writer.paths_for(stage, variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json
    report = {
        "schema_version": 1,
        "record_type": "pybullet_robustness_gate_report",
        "stage": stage,
        "variant": variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_record_type": str(source.get("record_type", "")),
        "source_run_id": str(source.get("run_id", "")),
        "thresholds": asdict(config.thresholds),
        "metrics": dict(source["metrics"]),
        "training": dict(source["training"]),
        "holdout": dict(source["holdout"]),
        "gates": gates,
        "readiness": {
            "robustness_claim": robustness_claim,
            "passed_gates": sum(1 for gate in gates if gate["passed"]),
            "required_gates": len(gates),
        },
        "boundary_notes": list(BOUNDARY_NOTES),
        "artifacts": {
            "summary_json": str(output_path),
            "manifest_json": str(manifest_path),
        },
    }
    writer.write_summary(output_path, report)
    writer.write_manifest(
        manifest_path,
        build_artifact_manifest(
            subject_record_type=report["record_type"],
            run_id=run_id,
            stage=stage,
            variant=variant,
            lineage={
                "source_record_type": report["source_record_type"],
                "source_run_id": report["source_run_id"],
                "robustness_claim": robustness_claim,
            },
            inputs=[artifact_reference(config.input_report, role="pybullet_holdout_report")],
            outputs=[artifact_reference(output_path, role="pybullet_robustness_gate_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _gates(
    source: dict[str, Any],
    thresholds: PyBulletRobustnessThresholds,
) -> list[dict[str, Any]]:
    readiness = source["readiness"]
    training = source["training"]
    holdout = source["holdout"]
    metrics = source["metrics"]
    return [
        _gate(
            "all_checkpoints_evaluated",
            bool(readiness.get("all_checkpoints_evaluated", False)),
            True,
            bool(readiness.get("all_checkpoints_evaluated", False)),
        ),
        _minimum_gate(
            "training_seed_count",
            int(training.get("seed_count", 0)),
            thresholds.min_training_seed_count,
        ),
        _minimum_gate(
            "min_total_timesteps",
            int(training.get("min_total_timesteps", 0)),
            thresholds.min_total_timesteps,
        ),
        _gate(
            "reserved_final_holdout",
            str(holdout.get("kind", "")),
            "reserved_final",
            str(holdout.get("kind", "")) == "reserved_final",
        ),
        _gate(
            "reserved_final_holdout_range",
            {
                "seed_start": int(holdout.get("seed_start", -1)),
                "seed_end": int(holdout.get("seed_end", -1)),
            },
            {
                "seed_start": FINAL_HOLDOUT_SEED_START,
                "seed_end": FINAL_HOLDOUT_SEED_END,
            },
            int(holdout.get("seed_start", -1)) == FINAL_HOLDOUT_SEED_START
            and int(holdout.get("seed_end", -1)) == FINAL_HOLDOUT_SEED_END,
        ),
        _minimum_gate(
            "holdout_episodes_per_checkpoint",
            int(holdout.get("episodes_per_checkpoint", 0)),
            thresholds.min_holdout_episodes_per_checkpoint,
        ),
        _minimum_gate(
            "worst_success_rate",
            _rate(metrics, "worst_success_rate"),
            thresholds.min_success_rate,
        ),
        _maximum_gate(
            "max_collision_rate",
            _rate(metrics, "max_collision_rate"),
            thresholds.max_collision_rate,
        ),
        _maximum_gate(
            "max_timeout_rate",
            _rate(metrics, "max_timeout_rate"),
            thresholds.max_timeout_rate,
        ),
        _minimum_gate(
            "worst_average_minimum_safety_distance",
            _non_negative_metric(metrics, "worst_average_minimum_safety_distance"),
            thresholds.min_average_minimum_safety_distance,
        ),
    ]


def _load_source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("input report must be a JSON object")
    if source.get("record_type") != "pybullet_multi_seed_checkpoint_holdout_report":
        raise ValueError("input must be a pybullet_multi_seed_checkpoint_holdout_report")
    for key in ("readiness", "training", "holdout", "metrics"):
        if not isinstance(source.get(key), dict):
            raise ValueError(f"input report must contain {key}")
    return source


def _minimum_gate(name: str, actual: float | int, minimum: float | int) -> dict[str, Any]:
    return _gate(name, actual, f">={minimum}", actual >= minimum)


def _maximum_gate(name: str, actual: float | int, maximum: float | int) -> dict[str, Any]:
    return _gate(name, actual, f"<={maximum}", actual <= maximum)


def _gate(name: str, actual: Any, expected: Any, passed: bool) -> dict[str, Any]:
    return {"name": name, "actual": actual, "expected": expected, "passed": bool(passed)}


def _finite(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping.get(key, 0.0))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _rate(mapping: dict[str, Any], key: str) -> float:
    value = _finite(mapping, key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return value


def _non_negative_metric(mapping: dict[str, Any], key: str) -> float:
    value = _finite(mapping, key)
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _config_hash(source: dict[str, Any], thresholds: PyBulletRobustnessThresholds) -> str:
    material = json.dumps(
        {"source_run_id": source.get("run_id", ""), "thresholds": asdict(thresholds)},
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
