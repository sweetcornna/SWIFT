from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_pybullet_multi_seed_training_script_dry_run_validates_plan(tmp_path: Path) -> None:
    output = tmp_path / "training.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_multi_seed_training.py",
            "--training-config",
            "configs/training_pybullet_randomized.yaml",
            "--seeds",
            "8",
            "9",
            "10",
            "--total-timesteps",
            "100000",
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "seeds=8,9,10" in result.stdout
    assert "total_timesteps=100000" in result.stdout
    assert not output.exists()


def test_pybullet_multi_seed_checkpoint_eval_script_dry_run_validates_input(tmp_path: Path) -> None:
    training_report = tmp_path / "training.json"
    training_report.write_text(
        json.dumps(
            {
                "record_type": "pybullet_multi_seed_training_report",
                "readiness": {"all_seeds_completed": True},
                "seed_runs": [{"seed": 8}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "holdout.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_multi_seed_checkpoint_eval.py",
            "--input",
            str(training_report),
            "--training-config",
            "configs/training_pybullet_randomized.yaml",
            "--episodes",
            "100",
            "--holdout-seed",
            "1000000",
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "episodes=100" in result.stdout
    assert "holdout_seed=1000000" in result.stdout
    assert not output.exists()


def test_pybullet_robustness_gate_script_writes_report_and_fails_on_reject(tmp_path: Path) -> None:
    source = tmp_path / "holdout.json"
    source.write_text(
        json.dumps(
            {
                "record_type": "pybullet_multi_seed_checkpoint_holdout_report",
                "run_id": "holdout-run",
                "stage": "stage1",
                "readiness": {"all_checkpoints_evaluated": True},
                "training": {"seed_count": 3, "min_total_timesteps": 100000},
                "holdout": {"seed_start": 1000000, "episodes_per_checkpoint": 100},
                "metrics": {
                    "worst_success_rate": 0.94,
                    "max_collision_rate": 0.0,
                    "max_timeout_rate": 0.05,
                    "worst_average_minimum_safety_distance": 0.10,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_robustness_gate.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--fail-on-reject",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["readiness"]["robustness_claim"] is False
    assert "robustness_claim=False" in result.stdout
