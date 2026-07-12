"""Configuration and immutable artifact contracts for stabilized hover."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from swift.experiments.artifacts import file_sha256

ACTION_PROFILE = "ct_att_yawrate_v1"
RESET_PROFILE = "robust_uniform_v1"
REWARD_PROFILE = "hover_kin_safe_v1"
NORMALIZATION_PROFILE = "train_rms_physical12_v1"
SELECTION_PROFILE = "heldout_robust_lexicographic_v1"
MODEL_NAMES = ("best_model.zip", "robust_best_model.zip", "final_model.zip")
PUBLICATION_MODEL_NAMES = ("robust_best_model.zip", "final_model.zip")


@dataclass(frozen=True)
class HoverTrainingConfig:
    total_timesteps: int = 100_000
    eval_frequency: int = 10_000
    eval_episodes: int = 20
    validation_cases: int = 100
    seed: int = 1
    device: str = "cpu"
    action_profile: str = ACTION_PROFILE
    initial_state_profile: str = RESET_PROFILE
    reward_profile: str = REWARD_PROFILE
    observation_normalization: str = NORMALIZATION_PROFILE
    checkpoint_selection: str = SELECTION_PROFILE

    def __post_init__(self) -> None:
        for name in ("total_timesteps", "eval_frequency", "eval_episodes", "validation_cases"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.eval_frequency > self.total_timesteps:
            raise ValueError("eval_frequency must not exceed total_timesteps")
        if not 0 <= int(self.seed) <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32-1]")
        if not str(self.device).strip():
            raise ValueError("device must not be empty")
        expected = {
            "action_profile": ACTION_PROFILE,
            "initial_state_profile": RESET_PROFILE,
            "reward_profile": REWARD_PROFILE,
            "observation_normalization": NORMALIZATION_PROFILE,
            "checkpoint_selection": SELECTION_PROFILE,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"stabilized hover requires {name}={value}")
        if self.total_timesteps == 100_000 and (self.eval_frequency != 10_000 or self.validation_cases != 100):
            raise ValueError("scientific 100000-step runs require eval_frequency=10000 and validation_cases=100")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HoverTrainingConfig":
        return cls(**{key: mapping[key] for key in cls.__dataclass_fields__ if key in mapping})


def load_hover_training_config(path: str | Path) -> HoverTrainingConfig:
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("hover configuration root must be a mapping")
    section = data.get("hover", data)
    if not isinstance(section, Mapping):
        raise ValueError("hover configuration section must be a mapping")
    return HoverTrainingConfig.from_mapping(section)


class HoverArtifactError(ValueError):
    pass


def load_bound_training_summary(
    run_dir: str | Path, *, required_models: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Validate profile/source hashes and exact bindings for requested or present models.

    By default, every recognized model archive actually present in ``run_dir`` is
    required.  Callers validating a complete training run may pass ``MODEL_NAMES``;
    callers selecting one model may pass a one-item tuple.  This keeps curated
    publications valid when they intentionally omit ``best_model.zip`` without
    weakening explicit full-run validation.
    """
    root = Path(run_dir).resolve()
    summary_path = root / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HoverArtifactError(f"cannot read training summary: {error}") from error
    if summary.get("schema_version") != 5:
        raise HoverArtifactError("ct_att_yawrate_v1 requires training summary schema 5")
    environment = summary.get("environment", {})
    if environment.get("action") != ACTION_PROFILE or environment.get("observation_shape") != [1, 72] or environment.get("action_shape") != [1, 4]:
        raise HoverArtifactError("training environment action/space contract mismatch")
    required_profiles = {
        ("initial_state_distribution", "name"): RESET_PROFILE,
        ("reward_profile", "name"): REWARD_PROFILE,
        ("observation_normalization", "name"): NORMALIZATION_PROFILE,
        ("checkpoint_selection", "name"): SELECTION_PROFILE,
        ("action_profile", "name"): ACTION_PROFILE,
    }
    for (section, key), expected in required_profiles.items():
        if not isinstance(summary.get(section), Mapping) or summary[section].get(key) != expected:
            raise HoverArtifactError(f"training {section} binding mismatch")
    artifacts = summary.get("artifacts")
    hashes = artifacts.get("sha256") if isinstance(artifacts, Mapping) else None
    bindings = summary.get("model_normalization_bindings")
    if not isinstance(hashes, Mapping) or not isinstance(bindings, Mapping):
        raise HoverArtifactError("artifact hash or normalization bindings are missing")
    for key in ("action_profile", "source_hashes"):
        basename = artifacts.get(key)
        if not isinstance(basename, str) or Path(basename).name != basename:
            raise HoverArtifactError(f"{key} must be a local artifact basename")
        path = root / basename
        if not path.is_file() or hashes.get(key) != file_sha256(path):
            raise HoverArtifactError(f"{key} hash binding mismatch")
    if required_models is None:
        required_models = tuple(name for name in MODEL_NAMES if (root / name).is_file())
        if not required_models:
            raise HoverArtifactError("no recognized model archives are present")
    else:
        required_models = tuple(required_models)
    if any(model_name not in MODEL_NAMES for model_name in required_models):
        raise HoverArtifactError("unknown required model binding")
    if not required_models:
        raise HoverArtifactError("required_models must not be empty")
    for model_name in required_models:
        kind = Path(model_name).stem
        binding = bindings.get(kind)
        if not isinstance(binding, Mapping) or binding.get("model") != model_name or binding.get("stats") != f"{kind}.obsnorm.npz":
            raise HoverArtifactError(f"missing exact normalization binding for {model_name}")
        for field, expected_name in (("model_sha256", model_name), ("stats_sha256", f"{kind}.obsnorm.npz")):
            target = root / expected_name
            if not target.is_file() or binding.get(field) != file_sha256(target):
                raise HoverArtifactError(f"{kind} {field} mismatch")
    return summary
