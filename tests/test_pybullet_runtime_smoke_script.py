import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_pybullet_runtime_smoke import _pixi_smoke_code, _pixi_training_env_smoke_code


def test_pybullet_runtime_smoke_pixi_code_can_enable_obstacles():
    code = _pixi_smoke_code(Path(r"D:\project\pybullet"), steps=1, enable_obstacles=True)

    assert "obstacles=True" in code
    assert "obstacles_enabled=True" in code


def test_pybullet_runtime_smoke_pixi_training_env_code_uses_swift_wrapper():
    code = _pixi_training_env_smoke_code(
        swift_root=Path(r"D:\project\SWIFT"),
        pybullet_root=Path(r"D:\project\pybullet"),
        steps=1,
        enable_obstacles=True,
    )

    assert "PyBulletVelocityTrainingEnv" in code
    assert "swift_obstacles=1" in code
    assert "runtime_contract=pybullet_velocity_training_compatibility" in code
    assert "nearest_obstacle_radius=" in code


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


def test_pybullet_runtime_training_env_smoke_script_runs_or_skips_cleanly():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_runtime_smoke.py",
            "--training-env",
            "--enable-obstacles",
            "--steps",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode == 2 and "PyBullet runtime unavailable" in (result.stdout + result.stderr):
        pytest.skip(result.stdout + result.stderr)

    assert result.returncode == 0, result.stderr
    assert "SWIFT PyBullet training env smoke: OK" in result.stdout
    assert "observation_dim=15" in result.stdout
    assert "runtime_contract=pybullet_velocity_training_compatibility" in result.stdout
    assert "swift_obstacles=1" in result.stdout
