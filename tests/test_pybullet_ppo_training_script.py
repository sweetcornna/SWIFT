import subprocess
import sys
from pathlib import Path


def test_pybullet_ppo_training_script_dry_run_validates_configs():
    result = subprocess.run(
        [sys.executable, "scripts/run_pybullet_ppo_training.py", "--dry-run"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT PyBullet PPO training config OK" in result.stdout
    assert "configs\\training.yaml" in result.stdout or "configs/training.yaml" in result.stdout
    assert "configs\\simulation.yaml" in result.stdout or "configs/simulation.yaml" in result.stdout
