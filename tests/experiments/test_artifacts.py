from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from swift.experiments.artifacts import (
    ExperimentArtifactConfig,
    ExperimentArtifactWriter,
    build_run_id,
    checkpoint_filename,
)
from swift.experiments.schema import ExperimentMetric


WINDOWS_UNSAFE_CHARS = set('<>:"/\\|?*')


def _assert_windows_safe(value: str) -> None:
    assert not any(char in WINDOWS_UNSAFE_CHARS for char in value)
    assert not any(char.isspace() for char in value)
    assert value == value.lower()


def test_write_episode_appends_parseable_jsonl_records(tmp_path: Path) -> None:
    writer = ExperimentArtifactWriter(ExperimentArtifactConfig(root=tmp_path))
    episode_path = tmp_path / "episodes" / "run.jsonl"

    writer.write_episode(episode_path, {"episode": 2, "return": 1.5})
    writer.write_episode(episode_path, {"episode": 3, "success": True})

    lines = episode_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"episode": 2, "return": 1.5},
        {"episode": 3, "success": True},
    ]


def test_write_summary_rejects_non_finite_numbers(tmp_path: Path) -> None:
    writer = ExperimentArtifactWriter()
    summary_path = tmp_path / "reports" / "summary.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        writer.write_summary(summary_path, {"loss": math.inf})

    assert not summary_path.exists()


def test_build_run_id_is_deterministic_and_windows_safe() -> None:
    started_at = datetime(2026, 6, 3, 4, 5, 6, tzinfo=timezone.utc)

    first = build_run_id(
        stage="Stage 1",
        variant="PPO/MLP Contract",
        seed=7,
        config_hash="ABCDEF123456",
        started_at_utc=started_at,
    )
    second = build_run_id(
        stage="Stage 1",
        variant="PPO/MLP Contract",
        seed=7,
        config_hash="ABCDEF123456",
        started_at_utc=started_at,
    )

    assert first == second
    assert first == "stage-1_ppo-mlp-contract_seed-7_cfg-abcdef12_20260603t040506z"
    _assert_windows_safe(first)


def test_checkpoint_filename_is_deterministic_zero_padded_and_windows_safe() -> None:
    filename = checkpoint_filename(
        variant="PPO/MLP Contract",
        run_id="stage-1_ppo-mlp-contract_seed-7_cfg-abcdef12_20260603t040506z",
        episode=12,
        step=345,
        label="Best Model",
        metric_name="success:rate",
        metric_value=0.8125,
    )

    assert (
        filename
        == "ppo-mlp-contract__stage-1_ppo-mlp-contract_seed-7_cfg-abcdef12_20260603t040506z"
        "__episode-000012__step-0000000345__best-model__success-rate-0.812500.ckpt"
    )
    _assert_windows_safe(filename)


def test_paths_for_keeps_artifacts_under_configured_roots(tmp_path: Path) -> None:
    config = ExperimentArtifactConfig(
        root=tmp_path / "outputs",
        episode_logs=tmp_path / "episode-root",
        experiment_reports=tmp_path / "report-root",
        checkpoints=tmp_path / "checkpoint-root",
    )
    paths = ExperimentArtifactWriter(config).paths_for(
        stage="Stage 1",
        variant="PPO/MLP Contract",
        run_id="stage-1_ppo-mlp-contract_seed-7_cfg-abcdef12_20260603t040506z",
    )

    assert paths.episode_jsonl == (
        config.episode_logs
        / "stage-1"
        / "ppo-mlp-contract"
        / "stage-1_ppo-mlp-contract_seed-7_cfg-abcdef12_20260603t040506z.jsonl"
    )
    assert paths.summary_json == (
        config.experiment_reports
        / "stage-1"
        / "ppo-mlp-contract"
        / "stage-1_ppo-mlp-contract_seed-7_cfg-abcdef12_20260603t040506z.json"
    )
    assert paths.checkpoint_dir == (
        config.checkpoints
        / "stage-1"
        / "ppo-mlp-contract"
        / "stage-1_ppo-mlp-contract_seed-7_cfg-abcdef12_20260603t040506z"
    )


def test_schema_includes_stage1_training_metrics() -> None:
    assert ExperimentMetric.TIMEOUT_RATE.value == "timeout_rate"
    assert ExperimentMetric.AVERAGE_PATH_SMOOTHNESS.value == "average_path_smoothness"
    assert ExperimentMetric.MINIMUM_SAFETY_DISTANCE.value == "minimum_safety_distance"
    assert ExperimentMetric.AVERAGE_EPISODE_RETURN.value == "average_episode_return"
    assert ExperimentMetric.MEAN_EPISODE_STEPS.value == "mean_episode_steps"
