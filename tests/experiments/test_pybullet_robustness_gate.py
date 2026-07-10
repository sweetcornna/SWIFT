from __future__ import annotations

import json
from pathlib import Path

import pytest


def _source_payload() -> dict:
    return {
        "record_type": "pybullet_multi_seed_checkpoint_holdout_report",
        "run_id": "holdout-run",
        "stage": "stage1",
        "readiness": {"all_checkpoints_evaluated": True},
        "training": {
            "seed_count": 3,
            "seeds": [8, 9, 10],
            "min_total_timesteps": 100000,
            "aggregate_total_timesteps": 300000,
        },
        "holdout": {
            "seed_start": 1000000,
            "seed_end": 1000099,
            "episodes_per_checkpoint": 100,
            "total_episodes": 300,
        },
        "metrics": {
            "worst_success_rate": 0.95,
            "max_collision_rate": 0.0,
            "max_timeout_rate": 0.05,
            "worst_average_minimum_safety_distance": 0.10,
        },
    }


def _write_source(path: Path, payload: dict | None = None) -> Path:
    path.write_text(json.dumps(payload or _source_payload()), encoding="utf-8")
    return path


def test_pybullet_robustness_gate_accepts_strict_worst_case_thresholds(tmp_path: Path) -> None:
    from swift.experiments.pybullet_robustness_gate import (
        PyBulletRobustnessGateConfig,
        run_pybullet_robustness_gate,
    )

    source = _write_source(tmp_path / "holdout.json")
    output = tmp_path / "gate.json"

    report = run_pybullet_robustness_gate(
        PyBulletRobustnessGateConfig(input_report=source, output=output)
    )

    assert report["record_type"] == "pybullet_robustness_gate_report"
    assert report["readiness"]["robustness_claim"] is True
    assert report["readiness"]["passed_gates"] == 8
    assert report["readiness"]["required_gates"] == 8
    assert all(gate["passed"] for gate in report["gates"])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert {item["role"] for item in manifest["inputs"]} == {"pybullet_holdout_report"}
    assert {item["role"] for item in manifest["outputs"]} == {"pybullet_robustness_gate_report"}


@pytest.mark.parametrize(
    ("mutator", "failed_gate"),
    [
        (lambda report: report["readiness"].update(all_checkpoints_evaluated=False), "all_checkpoints_evaluated"),
        (lambda report: report["training"].update(seed_count=2), "training_seed_count"),
        (lambda report: report["training"].update(min_total_timesteps=99999), "min_total_timesteps"),
        (lambda report: report["holdout"].update(episodes_per_checkpoint=99), "holdout_episodes_per_checkpoint"),
        (lambda report: report["metrics"].update(worst_success_rate=0.94), "worst_success_rate"),
        (lambda report: report["metrics"].update(max_collision_rate=0.01), "max_collision_rate"),
        (lambda report: report["metrics"].update(max_timeout_rate=0.06), "max_timeout_rate"),
        (
            lambda report: report["metrics"].update(worst_average_minimum_safety_distance=0.09),
            "worst_average_minimum_safety_distance",
        ),
    ],
)
def test_pybullet_robustness_gate_rejects_each_failed_requirement(
    tmp_path: Path,
    mutator,
    failed_gate: str,
) -> None:
    from swift.experiments.pybullet_robustness_gate import (
        PyBulletRobustnessGateConfig,
        run_pybullet_robustness_gate,
    )

    payload = _source_payload()
    mutator(payload)
    source = _write_source(tmp_path / f"{failed_gate}.json", payload)

    report = run_pybullet_robustness_gate(
        PyBulletRobustnessGateConfig(
            input_report=source,
            output=tmp_path / f"{failed_gate}-gate.json",
        )
    )

    gates = {gate["name"]: gate for gate in report["gates"]}
    assert gates[failed_gate]["passed"] is False
    assert report["readiness"]["robustness_claim"] is False


def test_pybullet_robustness_gate_rejects_wrong_record_type(tmp_path: Path) -> None:
    from swift.experiments.pybullet_robustness_gate import (
        PyBulletRobustnessGateConfig,
        run_pybullet_robustness_gate,
    )

    payload = _source_payload()
    payload["record_type"] = "pybullet_ppo_checkpoint_evaluation"

    with pytest.raises(ValueError, match="pybullet_multi_seed_checkpoint_holdout_report"):
        run_pybullet_robustness_gate(
            PyBulletRobustnessGateConfig(
                input_report=_write_source(tmp_path / "wrong.json", payload),
                output=tmp_path / "gate.json",
            )
        )


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("worst_success_rate", 1.01),
        ("max_collision_rate", -0.01),
        ("max_timeout_rate", 1.01),
        ("worst_average_minimum_safety_distance", -0.01),
    ],
)
def test_pybullet_robustness_gate_rejects_out_of_range_metrics(
    tmp_path: Path,
    metric: str,
    value: float,
) -> None:
    from swift.experiments.pybullet_robustness_gate import (
        PyBulletRobustnessGateConfig,
        run_pybullet_robustness_gate,
    )

    payload = _source_payload()
    payload["metrics"][metric] = value

    with pytest.raises(ValueError, match=metric):
        run_pybullet_robustness_gate(
            PyBulletRobustnessGateConfig(
                input_report=_write_source(tmp_path / f"{metric}.json", payload),
                output=tmp_path / "gate.json",
            )
        )
