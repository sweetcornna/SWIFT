from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import math

from swift.config import TrainingRunSettings, TrainingSettings
from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)
from swift.experiments.hca_training_runner import HCATrainingRunConfig, run_hca_training_smoke
from swift.experiments.ppo_training_runner import PPOTrainingRunConfig, run_ppo_training_smoke


REQUIRED_ABLATION_VARIANTS = ("ppo_mlp", "ppo_hca", "ppo_hca_apf")


@dataclass(frozen=True)
class TrainingAblationConfig:
    settings: TrainingSettings
    total_timesteps: int | None = None
    seed: int | None = None
    output: Path | None = None
    stage: str = "stage4"
    variant: str = "ablation"
    evidence_level: str = "cpu_smoke_ablation"

    def __post_init__(self) -> None:
        if self.total_timesteps is not None and self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "stage", str(self.stage).strip() or "stage4")
        object.__setattr__(self, "variant", str(self.variant).strip() or "ablation")
        object.__setattr__(self, "evidence_level", str(self.evidence_level).strip() or "cpu_smoke_ablation")


def run_training_ablation(config: TrainingAblationConfig) -> dict[str, Any]:
    writer = ExperimentArtifactWriter(config.settings.artifact)
    total_timesteps = config.total_timesteps or config.settings.run.total_timesteps
    seed = config.seed if config.seed is not None else config.settings.run.seed
    config_hash = _config_hash(
        config.settings,
        total_timesteps,
        seed,
        config.stage,
        config.variant,
        config.evidence_level,
    )
    run_id = build_run_id(
        stage=config.stage,
        variant=config.variant,
        seed=seed,
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(config.stage, config.variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if config.output is not None else paths.manifest_json

    variant_summaries = _run_required_variants(
        config=config,
        total_timesteps=total_timesteps,
        seed=seed,
        output_dir=output_path.parent / f"{output_path.stem}_variants",
    )
    variants = [_variant_record(summary) for summary in variant_summaries]
    ranking = _rank_variants(variants)
    for ranked in ranking:
        variants_by_name = {variant["variant"]: variant for variant in variants}
        variants_by_name[ranked["variant"]]["rank"] = ranked["rank"]

    report = {
        "schema_version": 1,
        "record_type": "training_ablation_report",
        "stage": config.stage,
        "variant": config.variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lineage": {
            "config_hash": config_hash,
            "training_goal": "compare_ppo_mlp_hca_hca_apf_smoke_variants",
            "evidence_level": config.evidence_level,
        },
        "total_timesteps": total_timesteps,
        "seed": seed,
        "required_variants": list(REQUIRED_ABLATION_VARIANTS),
        "ranking_formula": {
            "score": "100*success_rate - 100*collision_rate - 10*timeout_rate + 0.01*average_episode_return",
            "sort": "score_desc_variant_asc",
            "convergence_claim": False,
        },
        "best_variant": ranking[0]["variant"],
        "ranking": ranking,
        "variants": variants,
        "readiness": {
            "all_variants_completed": all(variant["completed"] for variant in variants),
            "completed_variants": sum(1 for variant in variants if variant["completed"]),
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
                artifact_reference(summary["artifacts"]["summary_json"], role=f"{summary['variant']}_summary")
                for summary in variant_summaries
            ],
            outputs=[artifact_reference(output_path, role="ablation_report")],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def score_ablation_metrics(metrics: dict[str, Any]) -> float:
    success_rate = _finite_metric(metrics, "success_rate")
    collision_rate = _finite_metric(metrics, "collision_rate")
    timeout_rate = _finite_metric(metrics, "timeout_rate")
    average_episode_return = _finite_metric(metrics, "average_episode_return")
    return (
        100.0 * success_rate
        - 100.0 * collision_rate
        - 10.0 * timeout_rate
        + 0.01 * average_episode_return
    )


def _run_required_variants(
    *,
    config: TrainingAblationConfig,
    total_timesteps: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mlp_settings = _variant_settings(config.settings, stage=config.stage, variant="ppo_mlp", seed=seed)
    mlp = run_ppo_training_smoke(
        PPOTrainingRunConfig(
            settings=mlp_settings,
            total_timesteps=total_timesteps,
            seed=seed,
            output=output_dir / "ppo_mlp.json",
        )
    )
    hca_settings = _variant_settings(config.settings, stage=config.stage, variant="ppo_hca", seed=seed)
    hca = run_hca_training_smoke(
        HCATrainingRunConfig(
            settings=hca_settings,
            total_timesteps=total_timesteps,
            seed=seed,
            output=output_dir / "ppo_hca.json",
            stage=config.stage,
            variant="ppo_hca",
        )
    )
    hca_apf_settings = _variant_settings(config.settings, stage=config.stage, variant="ppo_hca_apf", seed=seed)
    hca_apf = run_hca_training_smoke(
        HCATrainingRunConfig(
            settings=hca_apf_settings,
            total_timesteps=total_timesteps,
            seed=seed,
            output=output_dir / "ppo_hca_apf.json",
            stage=config.stage,
            variant="ppo_hca",
            enable_apf=True,
        )
    )
    return [mlp, hca, hca_apf]


def _variant_settings(settings: TrainingSettings, *, stage: str, variant: str, seed: int) -> TrainingSettings:
    return replace(
        settings,
        run=TrainingRunSettings(
            stage=stage,
            variant=variant,
            seed=seed,
            episodes=settings.run.episodes,
            total_timesteps=settings.run.total_timesteps,
        ),
    )


def _variant_record(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(summary["metrics"])
    score = score_ablation_metrics(metrics)
    return {
        "variant": str(summary["variant"]),
        "source_record_type": str(summary["record_type"]),
        "run_id": str(summary["run_id"]),
        "score": score,
        "completed": int(summary["training"]["updates"]) > 0
        and int(summary["training"]["total_timesteps"]) > 0,
        "metrics": metrics,
        "training": dict(summary["training"]),
        "artifacts": dict(summary["artifacts"]),
    }


def _rank_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        variants,
        key=lambda variant: (
            -float(variant["score"]),
            str(variant["variant"]),
        ),
    )
    return [
        {
            "rank": index,
            "variant": str(variant["variant"]),
            "score": float(variant["score"]),
            "metrics": dict(variant["metrics"]),
        }
        for index, variant in enumerate(ranked, start=1)
    ]


def _finite_metric(metrics: dict[str, Any], name: str) -> float:
    value = float(metrics.get(name, 0.0))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _config_hash(
    settings: TrainingSettings,
    total_timesteps: int,
    seed: int,
    stage: str,
    variant: str,
    evidence_level: str,
) -> str:
    material = repr((settings, total_timesteps, seed, stage, variant, evidence_level)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
