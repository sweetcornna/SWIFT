"""External-substrate bridge for SWIFT stabilized hover workflows."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    file_sha256,
)
from swift.hover.action_profiles import action_profile_metadata
from swift.hover.contracts import (
    ACTION_PROFILE,
    MODEL_NAMES,
    HoverArtifactError,
    HoverTrainingConfig,
    load_bound_training_summary,
)

TRAIN_SCRIPT = Path("scripts/train_drone_ppo.py")
EVAL_SCRIPT = Path("scripts/evaluate_drone_ppo.py")
VIZ_PACKAGE = Path("ppo_viz")


class HoverSubstrateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExternalHoverSubstrate:
    root: Path
    pixi_executable: Path

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        pixi = Path(self.pixi_executable)
        if not pixi.is_absolute():
            pixi = root / pixi
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "pixi_executable", pixi.resolve())

    def validate(self, *, require_runtime: bool = True) -> dict[str, str]:
        required = (TRAIN_SCRIPT, EVAL_SCRIPT, VIZ_PACKAGE, Path("scripts/action_profiles.py"))
        missing = [str(self.root / item) for item in required if not (self.root / item).exists()]
        if missing:
            raise HoverSubstrateError("external hover substrate is incomplete: " + ", ".join(missing))
        if require_runtime and not self.pixi_executable.is_file():
            raise HoverSubstrateError(f"Pixi executable not found: {self.pixi_executable}")
        return {item.as_posix(): file_sha256(self.root / item) for item in required if (self.root / item).is_file()}

    def source_hashes(self) -> dict[str, str]:
        files = (
            "scripts/action_profiles.py", "scripts/training_initial_states.py",
            "scripts/training_reward.py", "scripts/observation_normalization.py",
            "scripts/robust_validation.py", "scripts/train_drone_ppo.py",
            "scripts/evaluate_drone_ppo.py",
        )
        missing = [str(self.root / name) for name in files if not (self.root / name).is_file()]
        if missing:
            raise HoverSubstrateError("external hover source contract is incomplete: " + ", ".join(missing))
        return {name: file_sha256(self.root / name) for name in files}

    def run_pixi_python(self, arguments: Sequence[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        self.validate(require_runtime=True)
        command = [str(self.pixi_executable), "run", "python", *map(str, arguments)]
        completed = subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=timeout, check=False)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise HoverSubstrateError(f"external hover command failed ({completed.returncode}): {detail}")
        return completed


def run_training(substrate: ExternalHoverSubstrate, config: HoverTrainingConfig, output_root: Path, *, run_name: str | None = None, timeout: int | None = None) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    before = set(output_root.iterdir()) if output_root.is_dir() else set()
    arguments = [
        str(TRAIN_SCRIPT), "--action", ACTION_PROFILE,
        "--initial-state-profile", config.initial_state_profile,
        "--reward-profile", config.reward_profile,
        "--checkpoint-selection", config.checkpoint_selection,
        "--observation-normalization", config.observation_normalization,
        "--timesteps", str(config.total_timesteps), "--eval-freq", str(config.eval_frequency),
        "--eval-episodes", str(config.eval_episodes), "--validation-cases", str(config.validation_cases),
        "--seed", str(config.seed), "--device", config.device, "--output-root", str(output_root),
    ]
    if run_name:
        arguments.extend(("--run-name", run_name))
    substrate.run_pixi_python(arguments, timeout=timeout)
    run_dir = output_root / run_name if run_name else _single_new_directory(output_root, before)
    summary = load_bound_training_summary(run_dir)
    _verify_source_contract(substrate, summary)
    _write_swift_sidecars(substrate, run_dir, summary)
    return summary


def run_evaluation(substrate: ExternalHoverSubstrate, model_path: Path, output_path: Path, *, evaluation_seed: int = 0, cases: int = 100, device: str = "cpu", timeout: int | None = None) -> dict[str, Any]:
    model_path, output_path = Path(model_path).resolve(), Path(output_path).resolve()
    summary = load_bound_training_summary(model_path.parent, required_models=(model_path.name,))
    _verify_model_binding(summary, model_path)
    before = _snapshot_tree(model_path.parent)
    substrate.run_pixi_python([
        str(EVAL_SCRIPT), "--model", str(model_path), "--output", str(output_path),
        "--evaluation-seed", str(evaluation_seed), "--cases", str(cases), "--device", device,
        "--action", ACTION_PROFILE,
    ], timeout=timeout)
    if before != _snapshot_tree(model_path.parent):
        raise HoverSubstrateError("evaluation changed the input training run")
    report_path = output_path / "evaluation.json"
    if not report_path.is_file():
        raise HoverSubstrateError("evaluation did not produce evaluation.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_raw_physical_metrics(report, output_path / "evaluation.npz")
    _write_evaluation_manifest(model_path, output_path, report)
    return report


def run_visualization(substrate: ExternalHoverSubstrate, inputs: Sequence[Path], output_path: Path, *, html: bool = True, timeout: int | None = None) -> Path:
    sources = tuple(Path(item).resolve() for item in inputs)
    output = Path(output_path).resolve()
    if output.exists():
        raise HoverSubstrateError(f"visualization output already exists: {output}")
    before = {source: _snapshot_tree(source) for source in sources}
    arguments = ["-m", "ppo_viz", "build"]
    for source in sources:
        arguments.extend(("--input", str(source)))
    arguments.extend(("--output", str(output), "--html" if html else "--no-html"))
    substrate.run_pixi_python(arguments, timeout=timeout)
    if any(before[source] != _snapshot_tree(source) for source in sources):
        raise HoverSubstrateError("visualization changed an input artifact directory")
    if not (output / "manifest.json").is_file():
        raise HoverSubstrateError("visualization did not publish its read-only manifest")
    return output


def load_external_module(substrate: ExternalHoverSubstrate, relative_path: str, module_name: str) -> Any:
    """Load an external helper explicitly without mutating global import search paths."""
    path = substrate.root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise HoverSubstrateError(f"cannot load external helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _single_new_directory(root: Path, before: set[Path]) -> Path:
    after = set(root.iterdir()) if root.is_dir() else set()
    created = [path for path in after - before if path.is_dir()]
    if len(created) != 1:
        raise HoverSubstrateError("could not uniquely identify external training run directory")
    return created[0]


def _verify_source_contract(substrate: ExternalHoverSubstrate, summary: dict[str, Any]) -> None:
    source = summary.get("source", {})
    expected = substrate.source_hashes()
    mapping = {
        "action_profile_sha256": "scripts/action_profiles.py",
        "reset_wrapper_sha256": "scripts/training_initial_states.py",
        "reward_wrapper_sha256": "scripts/training_reward.py",
        "observation_normalization_sha256": "scripts/observation_normalization.py",
        "robust_validation_sha256": "scripts/robust_validation.py",
        "training_script_sha256": "scripts/train_drone_ppo.py",
    }
    for summary_key, source_path in mapping.items():
        if source.get(summary_key) != expected[source_path]:
            raise HoverArtifactError(f"external source hash mismatch: {source_path}")


def _verify_model_binding(summary: dict[str, Any], model_path: Path) -> None:
    if model_path.name not in MODEL_NAMES:
        raise HoverArtifactError("model must be a bound best, robust-best, or final archive")
    binding = summary["model_normalization_bindings"][model_path.stem]
    if binding["model_sha256"] != file_sha256(model_path):
        raise HoverArtifactError("selected model hash does not match training binding")


def _validate_raw_physical_metrics(report: dict[str, Any], npz_path: Path) -> None:
    required = {"position_m", "altitude_m", "rpy_rad", "linear_velocity_mps", "motor_rpm", "motor_pwm_saturated"}
    import numpy as np
    with np.load(npz_path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise HoverArtifactError("evaluation archive is missing raw physical metrics")
        if any(archive[name].dtype.kind == "O" for name in required):
            raise HoverArtifactError("raw physical metrics may not use object arrays")
        numeric = required - {"motor_pwm_saturated"}
        if any(not np.all(np.isfinite(archive[name])) for name in numeric):
            raise HoverArtifactError("raw physical metrics contain non-finite values")
    environment = report.get("environment", {})
    if environment.get("action") != ACTION_PROFILE.upper() or environment.get("physics_hz") != 240 or environment.get("control_hz") != 30:
        raise HoverArtifactError("evaluation physical runtime contract mismatch")


def _write_swift_sidecars(substrate: ExternalHoverSubstrate, run_dir: Path, summary: dict[str, Any]) -> None:
    writer = ExperimentArtifactWriter()
    report_path = run_dir / "swift_training_report.json"
    manifest_path = run_dir / "swift_training_report.manifest.json"
    if report_path.exists() or manifest_path.exists():
        raise HoverSubstrateError("SWIFT training sidecars already exist")
    payload = {
        "schema_version": 1, "record_type": "swift_pybullet_hover_training_report",
        "run_id": summary["run_id"], "stage": "pybullet_hover", "variant": ACTION_PROFILE,
        "lineage": {"integration": "external_substrate", "substrate_root": str(substrate.root), "training_summary_sha256": file_sha256(run_dir / "summary.json")},
        "profiles": {"action": ACTION_PROFILE, "reset": summary["initial_state_distribution"]["name"], "reward": summary["reward_profile"]["name"], "normalization": summary["observation_normalization"]["name"], "selection": summary["checkpoint_selection"]["name"]},
        "action_profile": action_profile_metadata(),
        "artifacts": {"source_summary_json": str(run_dir / "summary.json"), "manifest_json": str(manifest_path)},
    }
    writer.write_summary(report_path, payload)
    writer.write_manifest(manifest_path, build_artifact_manifest(
        subject_record_type=payload["record_type"], run_id=summary["run_id"], stage="pybullet_hover", variant=ACTION_PROFILE,
        inputs=[artifact_reference(run_dir / "summary.json", role="external_training_summary")],
        outputs=[artifact_reference(report_path, role="swift_training_report")], lineage=payload["lineage"],
    ))


def _write_evaluation_manifest(model_path: Path, output_path: Path, report: dict[str, Any]) -> None:
    writer = ExperimentArtifactWriter()
    manifest = output_path / "swift_evaluation.manifest.json"
    writer.write_manifest(manifest, build_artifact_manifest(
        subject_record_type="swift_pybullet_hover_evaluation", run_id=str(report["evaluation_id"]),
        stage="pybullet_hover", variant=ACTION_PROFILE,
        inputs=[artifact_reference(model_path, role="bound_model"), artifact_reference(model_path.parent / f"{model_path.stem}.obsnorm.npz", role="bound_observation_normalizer")],
        outputs=[artifact_reference(output_path / "evaluation.json", role="evaluation_report"), artifact_reference(output_path / "evaluation.npz", role="raw_physical_metrics")],
        lineage={"evaluation_backend": "external_substrate_read_only"},
    ))


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int, str]]:
    return {path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, file_sha256(path)) for path in sorted(root.rglob("*")) if path.is_file()}
