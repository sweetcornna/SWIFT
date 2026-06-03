from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentArtifactConfig:
    root: Path = field(default_factory=lambda: Path("outputs"))
    episode_logs: Path = field(default_factory=lambda: Path("outputs") / "episodes")
    experiment_reports: Path = field(default_factory=lambda: Path("outputs") / "reports")
    checkpoints: Path = field(default_factory=lambda: Path("checkpoints") / "stage1")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _path(self.root))
        object.__setattr__(self, "episode_logs", _path(self.episode_logs))
        object.__setattr__(self, "experiment_reports", _path(self.experiment_reports))
        object.__setattr__(self, "checkpoints", _path(self.checkpoints))


@dataclass(frozen=True)
class ExperimentArtifactPaths:
    episode_jsonl: Path
    summary_json: Path
    checkpoint_dir: Path


class ExperimentArtifactWriter:
    def __init__(self, config: ExperimentArtifactConfig | None = None) -> None:
        self.config = config or ExperimentArtifactConfig()

    def paths_for(self, stage: str, variant: str, run_id: str) -> ExperimentArtifactPaths:
        stage_slug = _safe_component(stage)
        variant_slug = _safe_component(variant)
        run_slug = _safe_component(run_id)
        return ExperimentArtifactPaths(
            episode_jsonl=self.config.episode_logs / stage_slug / variant_slug / f"{run_slug}.jsonl",
            summary_json=self.config.experiment_reports / stage_slug / variant_slug / f"{run_slug}.json",
            checkpoint_dir=self.config.checkpoints / stage_slug / variant_slug / run_slug,
        )

    def write_episode(self, path: str | Path, record: dict[str, Any]) -> None:
        serialized = json.dumps(record, allow_nan=False, sort_keys=True)
        episode_path = Path(path)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        with episode_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{serialized}\n")

    def write_summary(self, path: str | Path, summary: dict[str, Any]) -> None:
        serialized = json.dumps(summary, allow_nan=False, sort_keys=True)
        summary_path = Path(path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{serialized}\n", encoding="utf-8")


def build_run_id(
    stage: str,
    variant: str,
    seed: int,
    config_hash: str,
    started_at_utc: datetime,
) -> str:
    timestamp = _utc_timestamp(started_at_utc)
    config_slug = _safe_component(str(config_hash))[:8] or "none"
    return (
        f"{_safe_component(stage)}_{_safe_component(variant)}_"
        f"seed-{int(seed)}_cfg-{config_slug}_{timestamp}"
    )


def checkpoint_filename(
    variant: str,
    run_id: str,
    episode: int,
    step: int,
    label: str,
    metric_name: str,
    metric_value: float,
) -> str:
    if episode < 0:
        raise ValueError("episode must be non-negative")
    if step < 0:
        raise ValueError("step must be non-negative")
    metric = float(metric_value)
    if not math.isfinite(metric):
        raise ValueError("metric_value must be finite")

    return (
        f"{_safe_component(variant)}__{_safe_component(run_id)}__"
        f"episode-{int(episode):06d}__step-{int(step):010d}__"
        f"{_safe_component(label)}__{_safe_component(metric_name)}-{metric:.6f}.ckpt"
    )


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _utc_timestamp(value: datetime) -> str:
    timestamp = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).strftime("%Y%m%dt%H%M%Sz")


def _safe_component(value: object) -> str:
    raw = str(value).strip().lower()
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in raw)
    sanitized = re.sub("-+", "-", sanitized)
    sanitized = re.sub("_+", "_", sanitized)
    return sanitized.strip("-_") or "unnamed"
