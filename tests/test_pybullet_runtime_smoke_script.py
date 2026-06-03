import subprocess
import sys
from pathlib import Path

import pytest


def test_pybullet_runtime_smoke_script_runs_or_skips_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/run_pybullet_runtime_smoke.py", "--steps", "1"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode == 2 and "PyBullet runtime unavailable" in (result.stdout + result.stderr):
        pytest.skip(result.stdout + result.stderr)

    assert result.returncode == 0, result.stderr
    assert "SWIFT PyBullet runtime smoke: OK" in result.stdout
    assert "observation_dim=15" in result.stdout
