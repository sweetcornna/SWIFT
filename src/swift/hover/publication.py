"""Curate immutable robust-hover evidence without copying trajectory bulk."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from swift.experiments.artifacts import file_sha256
from swift.hover.contracts import ACTION_PROFILE, PUBLICATION_MODEL_NAMES, load_bound_training_summary


EXPECTED_ACCEPTANCE_CHECKS = frozenset({
    "complete_final_windows",
    "horizon_errors",
    "maximum_altitude",
    "non_finite",
    "p95_mae",
    "p95_rmse",
    "p95_std",
    "safety_completion",
    "unsafe",
    "worst_case_mae",
})
EXPECTED_ACCEPTANCE_THRESHOLDS = {
    "complete_final_window_count": 100,
    "horizon_error_count": 0,
    "maximum_altitude_m_max": 2.0,
    "non_finite_count": 0,
    "p95_mae_m_max": 0.1,
    "p95_rmse_m_max": 0.12,
    "p95_std_m_max": 0.08,
    "safety_completion_count": 100,
    "unsafe_count": 0,
    "worst_case_mae_m_max": 0.2,
}


class HoverPublicationError(ValueError):
    pass


def compact_evaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Retain reproducibility, aggregate evidence, and verdict; omit per-case trajectories."""
    required = ("schema_version", "evaluation_id", "model_kind", "model_sha256_before", "normalization", "evaluation_seed", "case_count", "environment", "sampling", "safety_definition", "aggregates", "acceptance_thresholds", "acceptance_checks", "harness_validity_checks", "evaluation_valid", "verdict")
    missing = [name for name in required if name not in report]
    if missing:
        raise HoverPublicationError("evaluation report is missing: " + ", ".join(missing))
    compact = {name: report[name] for name in required}
    compact["record_type"] = "swift_compact_robust_hover_evaluation"
    compact["source_training_summary_sha256"] = report.get("source_training_summary_sha256")
    compact["runtime_versions"] = report.get("runtime_versions")
    return compact


def publish_120k_runs(
    training_runs: Sequence[Path], evaluation_dirs: Sequence[Path], destination: Path
) -> dict[str, Any]:
    if len(training_runs) != 3 or len(evaluation_dirs) != 3:
        raise HoverPublicationError("publication requires exactly three training runs and evaluations")
    destination = Path(destination).resolve()
    if destination.exists():
        raise HoverPublicationError(f"publication destination already exists: {destination}")
    destination.mkdir(parents=True)
    seeds: list[dict[str, Any]] = []
    try:
        for training_value, evaluation_value in zip(training_runs, evaluation_dirs, strict=True):
            training = Path(training_value).resolve()
            evaluation = Path(evaluation_value).resolve()
            summary = load_bound_training_summary(training, required_models=PUBLICATION_MODEL_NAMES)
            seed = int(summary["training"]["train_seed"])
            if summary["training"]["requested_timesteps"] != 120_000:
                raise HoverPublicationError(f"seed {seed} is not a requested 120k run")
            if seed in {item["seed"] for item in seeds}:
                raise HoverPublicationError(f"duplicate training seed: {seed}")
            report_path = evaluation / "evaluation.json"
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise HoverPublicationError(f"seed {seed} evaluation cannot be read: {error}") from error
            robust_binding = summary["model_normalization_bindings"]["robust_best_model"]
            normalization = report.get("normalization", {})
            harness_checks = report.get("harness_validity_checks", {})
            acceptance_checks = report.get("acceptance_checks", {})
            acceptance_thresholds = report.get("acceptance_thresholds", {})
            expected_summary_hash = file_sha256(training / "summary.json")
            if (
                report.get("model_kind") != "robust_best_model"
                or report.get("verdict") != "accepted"
                or report.get("evaluation_valid") is not True
                or report.get("case_count") != 100
                or not isinstance(harness_checks, Mapping)
                or harness_checks.get("full_sampling_contract") is not True
                or not isinstance(acceptance_checks, Mapping)
                or set(acceptance_checks) != EXPECTED_ACCEPTANCE_CHECKS
                or any(value is not True for value in acceptance_checks.values())
                or acceptance_thresholds != EXPECTED_ACCEPTANCE_THRESHOLDS
            ):
                raise HoverPublicationError(f"seed {seed} evaluation is not a valid accepted 100-case robust-best evaluation")
            if (
                report.get("model_sha256_before") != robust_binding["model_sha256"]
                or report.get("model_sha256_after") != robust_binding["model_sha256"]
                or not isinstance(normalization, Mapping)
                or normalization.get("stats_sha256_before") != robust_binding["stats_sha256"]
                or normalization.get("stats_sha256_after") != robust_binding["stats_sha256"]
                or report.get("source_training_summary_sha256") != expected_summary_hash
            ):
                raise HoverPublicationError(f"seed {seed} evaluation/model binding mismatch")
            seed_dir = destination / f"seed-{seed}"
            seed_dir.mkdir()
            names = [
                *PUBLICATION_MODEL_NAMES,
                *(f"{Path(name).stem}.obsnorm.npz" for name in PUBLICATION_MODEL_NAMES),
                "action_profile.json", "source_hashes.json", "summary.json",
            ]
            for name in names:
                source = training / name
                if not source.is_file():
                    raise HoverPublicationError(f"missing publication input: {source}")
                shutil.copy2(source, seed_dir / name)
            compact_path = seed_dir / "evaluation.compact.json"
            compact_path.write_text(json.dumps(compact_evaluation(report), indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
            seeds.append({
                "seed": seed,
                "run_name": summary["run_name"],
                "robust_best_timestep": summary["checkpoint_selection"]["best_timestep"],
                "robust_best_score": summary["checkpoint_selection"]["best_score"],
                "evaluation_id": report["evaluation_id"],
                "verdict": report["verdict"],
            })
        if sorted(item["seed"] for item in seeds) != [1, 2, 3]:
            raise HoverPublicationError("publication requires training seeds 1, 2, and 3")
        files = []
        for path in sorted(item for item in destination.rglob("*") if item.is_file()):
            files.append({"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        manifest = {
            "schema_version": 1,
            "record_type": "swift_robust_hover_model_publication",
            "action_profile": ACTION_PROFILE,
            "requested_timesteps": 120_000,
            "scope": "single-drone headless PyBullet stabilized hover; not navigation or real-flight safety evidence",
            "excluded": ["evaluation.npz", "evaluations.npz", "robust_evaluations.npz", "validation_bank.npz", "trajectory bulk", "external substrate source"],
            "seeds": sorted(seeds, key=lambda item: item["seed"]),
            "files": files,
        }
        (destination / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
