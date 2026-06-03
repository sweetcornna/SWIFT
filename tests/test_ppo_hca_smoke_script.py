import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_ppo_hca_smoke_script_writes_summary(tmp_path: Path):
    output = tmp_path / "hca_smoke.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ppo_hca_smoke.py",
            "--total-timesteps",
            "64",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["record_type"] == "ppo_hca_training_smoke"
    assert summary["variant"] == "ppo_hca"
    assert summary["training"]["updates"] >= 1
    assert f"SWIFT PPO HCA smoke written: {output}" in result.stdout


def test_ppo_hca_smoke_script_can_enable_apf_fusion(tmp_path: Path):
    output = tmp_path / "hca_apf_smoke.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ppo_hca_smoke.py",
            "--enable-apf",
            "--total-timesteps",
            "64",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["variant"] == "ppo_hca_apf"
    assert summary["network"]["apf_enabled"] is True
    assert f"SWIFT PPO HCA smoke written: {output}" in result.stdout
