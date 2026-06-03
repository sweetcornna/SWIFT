from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import math

from swift.envs import SimpleAvoidanceSettings
from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)
from swift.experiments.ppo_checkpoint_evaluator import (
    PPOCheckpointEvaluationConfig,
    run_ppo_checkpoint_evaluation,
)


@dataclass(frozen=True)
class MultiSeedCheckpointEvaluationConfig:
    input_report: Path
    output: Path | None = None
    holdout_seeds: tuple[int, ...] = (10000, 11000, 12000)
    episodes: int = 3
    environment: SimpleAvoidanceSettings = field(default_factory=SimpleAvoidanceSettings)
    stage: str = "stage4"
    variant: str = "multi_seed_checkpoint_holdout"
    evidence_level: str = "long_training_convergence"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_report", Path(self.input_report))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        holdout_seeds = tuple(int(seed) for seed in self.holdout_seeds)
        if not holdout_seeds:
            raise ValueError("holdout_seeds must contain at least one seed")
        if any(seed < 0 for seed in holdout_seeds):
            raise ValueError("holdout_seeds must be non-negative")
        if len(set(holdout_seeds)) != len(holdout_seeds):
            raise ValueError("holdout_seeds must be unique")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        object.__setattr__(self, "holdout_seeds", holdout_seeds)
        object.__setattr__(self, "episodes", int(self.episodes))
        object.__setattr__(self, "stage", str(self.stage).strip() or "stage4")
        object.__setattr__(self, "variant", str(self.variant).strip() or "multi_seed_checkpoint_holdout")
        object.__setattr__(self, "evidence_level", str(self.evidence_level).strip() or "long_training_convergence")


def run_multi_seed_checkpoint_evaluation(config: MultiSeedCheckpointEvaluationConfig) -> dict[str, Any]:
    source = _load_source(config.input_report)
    training_seeds = tuple(int(seed) for seed in source.get("seeds", ()))
    _validate_holdout_seeds(training_seeds, config.holdout_seeds)
    config_hash = _config_hash(config, source)
    run_id = build_run_id(
        stage=config.stage,
        variant=config.variant,
        seed=min(config.holdout_seeds),
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    writer = ExperimentArtifactWriter()
    paths = writer.paths_for(config.stage, config.variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json
    evaluation_dir = output_path.parent / f"{output_path.stem}_checkpoint_evals"
    variants = _evaluate_variants(source, config, evaluation_dir)
    ranking = _rank_variants(variants)
    best_variant = ranking[0]["variant"] if ranking else ""
    report = {
        "schema_version": 1,
        "record_type": "multi_seed_checkpoint_holdout_report",
        "stage": config.stage,
        "variant": config.variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_record_type": str(source.get("record_type", "")),
        "source_run_id": str(source.get("run_id", "")),
        "lineage": {
            "config_hash": config_hash,
            "source_evidence_level": _evidence_level(source),
            "evidence_level": config.evidence_level,
            "source_runner": "swift.experiments.multi_seed_training_runner.run_multi_seed_training",
            "evaluator": "swift.experiments.ppo_checkpoint_evaluator.run_ppo_checkpoint_evaluation",
        },
        "training_seeds": list(training_seeds),
        "holdout_seeds": list(config.holdout_seeds),
        "seed_count": len(training_seeds),
        "holdout_seed_count": len(config.holdout_seeds),
        "episodes_per_checkpoint": config.episodes,
        "required_variants": list(source.get("required_variants", [])),
        "best_variant": best_variant,
        "ranking": ranking,
        "variants": variants,
        "readiness": {
            "all_checkpoints_evaluated": all(variant["completed"] for variant in variants),
            "training_seed_count": len(training_seeds),
            "holdout_seed_count": len(config.holdout_seeds),
            "convergence_claim": False,
            "evidence_level": config.evidence_level,
        },
        "artifacts": {
            "summary_json": str(output_path),
            "manifest_json": str(manifest_path),
            "evaluation_dir": str(evaluation_dir),
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
            lineage=report["lineage"],
            inputs=[artifact_reference(config.input_report, role="source_multi_seed_training_report")],
            outputs=[artifact_reference(output_path, role="multi_seed_checkpoint_holdout_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _load_source(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("input report must be a JSON object")
    if source.get("record_type") != "multi_seed_training_report":
        raise ValueError("input report record_type must be multi_seed_training_report")
    return source


def _validate_holdout_seeds(training_seeds: tuple[int, ...], holdout_seeds: tuple[int, ...]) -> None:
    if set(training_seeds) & set(holdout_seeds):
        raise ValueError("holdout_seeds must be disjoint from training seeds")


def _evaluate_variants(
    source: dict[str, Any],
    config: MultiSeedCheckpointEvaluationConfig,
    evaluation_dir: Path,
) -> list[dict[str, Any]]:
    variants = source.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("input report variants must be a list")
    return [
        _evaluate_variant(variant, config, evaluation_dir)
        for variant in variants
        if isinstance(variant, dict)
    ]


def _evaluate_variant(
    source_variant: dict[str, Any],
    config: MultiSeedCheckpointEvaluationConfig,
    evaluation_dir: Path,
) -> dict[str, Any]:
    variant_name = str(source_variant.get("variant", "unknown"))
    seed_metrics = source_variant.get("seed_metrics", [])
    if not isinstance(seed_metrics, list):
        raise ValueError("variant seed_metrics must be a list")
    checkpoint_evaluations = []
    for seed_metric in seed_metrics:
        if not isinstance(seed_metric, dict):
            continue
        checkpoint_evaluations.extend(_evaluate_seed_metric(variant_name, seed_metric, config, evaluation_dir))
    metrics = _aggregate_metrics(checkpoint_evaluations)
    source_training = dict(source_variant.get("training", {}))
    aggregate = {
        "variant": variant_name,
        "completed": len(checkpoint_evaluations) == len(seed_metrics) * len(config.holdout_seeds),
        "training_seed_count": len(seed_metrics),
        "seed_count": len(seed_metrics),
        "completed_seeds": len(seed_metrics),
        "holdout_seed_count": len(config.holdout_seeds),
        "holdout_checkpoint_count": len(checkpoint_evaluations),
        "metrics": metrics,
        "training": {
            "total_timesteps": int(source_training.get("total_timesteps", 0)),
            "updates": int(source_training.get("updates", 0)),
            "episodes_completed": int(source_training.get("episodes_completed", 0)),
        },
        "evaluation": {
            "episodes_completed": sum(int(item["episodes_requested"]) for item in checkpoint_evaluations),
            "checkpoint_count": len(checkpoint_evaluations),
        },
        "seed_metrics": _group_seed_metrics(checkpoint_evaluations),
        "checkpoint_evaluations": checkpoint_evaluations,
    }
    aggregate["aggregate_score"] = (
        100.0 * metrics["success_rate"]
        - 100.0 * metrics["collision_rate"]
        - 10.0 * metrics["timeout_rate"]
        + 0.01 * metrics["average_episode_return"]
    )
    return aggregate


def _evaluate_seed_metric(
    variant_name: str,
    seed_metric: dict[str, Any],
    config: MultiSeedCheckpointEvaluationConfig,
    evaluation_dir: Path,
) -> list[dict[str, Any]]:
    training_seed = int(seed_metric["seed"])
    checkpoint_path = Path(seed_metric.get("artifacts", {}).get("checkpoint_path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    records = []
    for holdout_seed in config.holdout_seeds:
        output_path = evaluation_dir / variant_name / f"train_seed_{training_seed}_holdout_seed_{holdout_seed}.json"
        summary = run_ppo_checkpoint_evaluation(
            PPOCheckpointEvaluationConfig(
                checkpoint_path=checkpoint_path,
                output=output_path,
                episodes=config.episodes,
                seed=holdout_seed,
                environment=config.environment,
            )
        )
        records.append(
            {
                "variant": variant_name,
                "training_seed": training_seed,
                "holdout_seed": holdout_seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_record_type": str(summary["checkpoint_record_type"]),
                "policy_family": str(summary["policy_family"]),
                "episodes_requested": int(summary["episodes_requested"]),
                "metrics": dict(summary["metrics"]),
                "training": dict(seed_metric.get("training", {})),
                "artifacts": {
                    "summary_json": str(output_path),
                    "checkpoint_path": str(checkpoint_path),
                },
            }
        )
    return records


def _aggregate_metrics(evaluations: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [dict(evaluation["metrics"]) for evaluation in evaluations]
    return {
        "success_rate": min(_finite_metric(metric, "success_rate") for metric in metrics),
        "collision_rate": max(_finite_metric(metric, "collision_rate") for metric in metrics),
        "timeout_rate": max(_finite_metric(metric, "timeout_rate") for metric in metrics),
        "average_episode_return": _mean(_finite_metric(metric, "average_episode_return") for metric in metrics),
        "average_steps": _mean(_finite_metric(metric, "average_steps") for metric in metrics),
        "average_minimum_safety_distance": _mean(
            _finite_metric(metric, "average_minimum_safety_distance") for metric in metrics
        ),
    }


def _group_seed_metrics(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "training_seed": int(evaluation["training_seed"]),
            "holdout_seed": int(evaluation["holdout_seed"]),
            "checkpoint_path": str(evaluation["checkpoint_path"]),
            "checkpoint_record_type": str(evaluation["checkpoint_record_type"]),
            "policy_family": str(evaluation["policy_family"]),
            "episodes_requested": int(evaluation["episodes_requested"]),
            "metrics": dict(evaluation["metrics"]),
            "training": dict(evaluation["training"]),
            "artifacts": dict(evaluation["artifacts"]),
        }
        for evaluation in evaluations
    ]


def _rank_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        variants,
        key=lambda variant: (-float(variant["aggregate_score"]), str(variant["variant"])),
    )
    return [
        {
            "rank": index,
            "variant": str(variant["variant"]),
            "aggregate_score": float(variant["aggregate_score"]),
            "success_rate": float(variant["metrics"]["success_rate"]),
            "collision_rate": float(variant["metrics"]["collision_rate"]),
            "timeout_rate": float(variant["metrics"]["timeout_rate"]),
        }
        for index, variant in enumerate(ranked, start=1)
    ]


def _evidence_level(source: dict[str, Any]) -> str:
    readiness = source.get("readiness", {})
    if isinstance(readiness, dict) and readiness.get("evidence_level"):
        return str(readiness["evidence_level"])
    lineage = source.get("lineage", {})
    if isinstance(lineage, dict) and lineage.get("evidence_level"):
        return str(lineage["evidence_level"])
    return "unknown"


def _finite_metric(metrics: dict[str, Any], name: str) -> float:
    value = float(metrics.get(name, 0.0))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _mean(values: Any) -> float:
    sequence = [float(value) for value in values]
    if not sequence:
        return 0.0
    return sum(sequence) / len(sequence)


def _config_hash(config: MultiSeedCheckpointEvaluationConfig, source: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "input_report": str(config.input_report),
            "source_run_id": source.get("run_id", ""),
            "holdout_seeds": list(config.holdout_seeds),
            "episodes": config.episodes,
            "environment": repr(config.environment),
            "stage": config.stage,
            "variant": config.variant,
            "evidence_level": config.evidence_level,
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
