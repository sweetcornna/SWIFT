import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_multi_seed_training_script_writes_report(tmp_path: Path):
    output = tmp_path / "multi_seed.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multi_seed_training.py",
            "--total-timesteps",
            "32",
            "--seeds",
            "0",
            "1",
            "--evidence-level",
            "long_training_convergence",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["record_type"] == "multi_seed_training_report"
    assert report["seed_count"] == 2
    assert report["readiness"]["evidence_level"] == "long_training_convergence"
    assert f"SWIFT multi-seed training written: {output}" in result.stdout
    assert "seed_count=2 variants=3" in result.stdout


def test_multi_seed_training_script_dry_run_validates_config():
    result = subprocess.run(
        [sys.executable, "scripts/run_multi_seed_training.py", "--dry-run"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT multi-seed training config OK" in result.stdout


def test_multi_seed_training_script_defaults_to_cpu_smoke_evidence_level(tmp_path: Path):
    output = tmp_path / "multi_seed.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multi_seed_training.py",
            "--total-timesteps",
            "32",
            "--seeds",
            "0",
            "1",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["readiness"]["evidence_level"] == "cpu_smoke_ablation"
