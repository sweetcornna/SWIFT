from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
import hashlib
import json
import math

from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)

if TYPE_CHECKING:
    from swift.config import TrainingSettings
from swift.experiments.training_ablation_runner import (
    REQUIRED_ABLATION_VARIANTS,
    TrainingAblationConfig,
    run_training_ablation,
    score_ablation_metrics,
)


@dataclass(frozen=True)
class MultiSeedTrainingConfig:
    settings: "TrainingSettings"
    seeds: tuple[int, ...] = (0, 1, 2)
    total_timesteps: int | None = None
    output: Path | None = None
    stage: str = "stage4"
    variant: str = "multi_seed_training"
    evidence_level: str = "long_training_convergence"

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("seeds must contain at least one seed")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if self.total_timesteps is not None and self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "stage", str(self.stage).strip() or "stage4")
        object.__setattr__(self, "variant", str(self.variant).strip() or "multi_seed_training")
        object.__setattr__(self, "evidence_level", str(self.evidence_level).strip() or "long_training_convergence")


def run_multi_seed_training(config: MultiSeedTrainingConfig) -> dict[str, Any]:
    writer = ExperimentArtifactWriter(config.settings.artifact)
    total_timesteps = config.total_timesteps or config.settings.run.total_timesteps
    config_hash = _config_hash(config, total_timesteps)
    run_id = build_run_id(
        stage=config.stage,
        variant=config.variant,
        seed=min(config.seeds),
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(config.stage, config.variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json
    seed_reports = _run_seed_reports(config, total_timesteps, output_path)
    variants = _aggregate_variants(seed_reports)
    ranking = _rank_variants(variants)
    best_variant = ranking[0]["variant"]

    report = {
        "schema_version": 1,
        "record_type": "multi_seed_training_report",
        "stage": config.stage,
        "variant": config.variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lineage": {
            "config_hash": config_hash,
            "source_runner": "swift.experiments.training_ablation_runner.run_training_ablation",
            "evidence_level": config.evidence_level,
        },
        "total_timesteps": total_timesteps,
        "seeds": list(config.seeds),
        "seed_count": len(config.seeds),
        "required_variants": list(REQUIRED_ABLATION_VARIANTS),
        "best_variant": best_variant,
        "ranking": ranking,
        "seed_reports": [_seed_report_record(report) for report in seed_reports],
        "variants": variants,
        "readiness": {
            "all_seeds_completed": all(_seed_completed(report) for report in seed_reports),
            "seed_count": len(config.seeds),
            "required_variants": len(REQUIRED_ABLATION_VARIANTS),
            "convergence_claim": False,
            "evidence_level": config.evidence_level,
        },
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
            stage=config.stage,
            variant=config.variant,
            lineage=report["lineage"],
            inputs=[
                artifact_reference(seed_report["artifacts"]["summary_json"], role=f"seed_{seed_report['seed']}_ablation_report")
                for seed_report in seed_reports
            ],
            outputs=[artifact_reference(output_path, role="multi_seed_training_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_seed_reports(
    config: MultiSeedTrainingConfig,
    total_timesteps: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    seed_dir = output_path.parent / f"{output_path.stem}_seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in config.seeds:
        reports.append(
            run_training_ablation(
                TrainingAblationConfig(
                    settings=config.settings,
                    total_timesteps=total_timesteps,
                    seed=seed,
                    output=seed_dir / f"seed_{seed}_ablation.json",
                    stage=config.stage,
                    variant="ablation",
                    evidence_level=config.evidence_level,
                )
            )
        )
    return reports


def _aggregate_variants(seed_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants_by_name: dict[str, list[dict[str, Any]]] = {variant: [] for variant in REQUIRED_ABLATION_VARIANTS}
    for seed_report in seed_reports:
        seed = int(seed_report["seed"])
        for variant in seed_report["variants"]:
            variants_by_name.setdefault(str(variant["variant"]), []).append(
                {
                    "seed": seed,
                    "completed": bool(variant["completed"]),
                    "score": float(variant["score"]),
                    "metrics": dict(variant["metrics"]),
                    "training": dict(variant["training"]),
                    "artifacts": dict(variant["artifacts"]),
                }
            )

    aggregates = []
    for variant_name in sorted(variants_by_name):
        seed_records = sorted(variants_by_name[variant_name], key=lambda item: int(item["seed"]))
        metrics = [record["metrics"] for record in seed_records]
        scores = [float(record["score"]) for record in seed_records]
        seed_total_timesteps = [int(record["training"].get("total_timesteps", 0)) for record in seed_records]
        seed_updates = [int(record["training"].get("updates", 0)) for record in seed_records]
        seed_episodes_completed = [int(record["training"].get("episodes_completed", 0)) for record in seed_records]
        aggregate = {
            "variant": variant_name,
            "seed_count": len(seed_records),
            "completed_seeds": sum(1 for record in seed_records if record["completed"]),
            "mean_score": _mean(scores),
            "worst_success_rate": min(_finite_metric(metric, "success_rate") for metric in metrics),
            "mean_success_rate": _mean(_finite_metric(metric, "success_rate") for metric in metrics),
            "max_collision_rate": max(_finite_metric(metric, "collision_rate") for metric in metrics),
            "max_timeout_rate": max(_finite_metric(metric, "timeout_rate") for metric in metrics),
            "mean_episode_return": _mean(_finite_metric(metric, "average_episode_return") for metric in metrics),
            "total_timesteps": min(seed_total_timesteps),
            "min_seed_total_timesteps": min(seed_total_timesteps),
            "aggregate_total_timesteps": sum(seed_total_timesteps),
            "episodes_completed": min(seed_episodes_completed),
            "min_seed_episodes_completed": min(seed_episodes_completed),
            "aggregate_episodes_completed": sum(seed_episodes_completed),
            "min_seed_updates": min(seed_updates),
            "aggregate_updates": sum(seed_updates),
            "seed_metrics": [
                {
                    "seed": int(record["seed"]),
                    "completed": bool(record["completed"]),
                    "score": float(record["score"]),
                    "metrics": dict(record["metrics"]),
                    "training": dict(record["training"]),
                    "artifacts": dict(record["artifacts"]),
                }
                for record in seed_records
            ],
        }
        aggregate["metrics"] = {
            "success_rate": aggregate["worst_success_rate"],
            "collision_rate": aggregate["max_collision_rate"],
            "timeout_rate": aggregate["max_timeout_rate"],
            "average_episode_return": aggregate["mean_episode_return"],
        }
        aggregate["training"] = {
            "total_timesteps": aggregate["min_seed_total_timesteps"],
            "updates": aggregate["min_seed_updates"],
            "episodes_completed": aggregate["min_seed_episodes_completed"],
        }
        aggregate["completed"] = aggregate["completed_seeds"] == aggregate["seed_count"]
        aggregate["aggregate_score"] = score_ablation_metrics(
            {
                "success_rate": aggregate["worst_success_rate"],
                "collision_rate": aggregate["max_collision_rate"],
                "timeout_rate": aggregate["max_timeout_rate"],
                "average_episode_return": aggregate["mean_episode_return"],
            }
        )
        aggregates.append(aggregate)
    return aggregates


def _rank_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        variants,
        key=lambda variant: (
            -float(variant["aggregate_score"]),
            str(variant["variant"]),
        ),
    )
    return [
        {
            "rank": index,
            "variant": str(variant["variant"]),
            "aggregate_score": float(variant["aggregate_score"]),
            "worst_success_rate": float(variant["worst_success_rate"]),
            "max_collision_rate": float(variant["max_collision_rate"]),
            "max_timeout_rate": float(variant["max_timeout_rate"]),
        }
        for index, variant in enumerate(ranked, start=1)
    ]


def _seed_report_record(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(report["seed"]),
        "run_id": str(report["run_id"]),
        "best_variant": str(report["best_variant"]),
        "completed": _seed_completed(report),
        "artifacts": dict(report["artifacts"]),
    }


def _seed_completed(report: dict[str, Any]) -> bool:
    readiness = report.get("readiness", {})
    return bool(isinstance(readiness, dict) and readiness.get("all_variants_completed", False))


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


def _config_hash(config: MultiSeedTrainingConfig, total_timesteps: int) -> str:
    material = repr(
        (
            config.settings,
            config.seeds,
            total_timesteps,
            config.stage,
            config.variant,
            config.evidence_level,
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
