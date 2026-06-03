import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_ppo_mlp_smoke_script_help_works_without_running_training():
    result = subprocess.run(
        [sys.executable, "scripts/run_ppo_mlp_smoke.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--total-timesteps" in result.stdout


def test_ppo_mlp_smoke_script_writes_training_summary(tmp_path: Path):
    pytest.importorskip("torch")
    output = tmp_path / "ppo_smoke.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ppo_mlp_smoke.py",
            "--total-timesteps",
            "128",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training"]["total_timesteps"] == 128
    assert payload["training"]["updates"] >= 1
    assert f"SWIFT PPO MLP smoke written: {output}" in result.stdout
