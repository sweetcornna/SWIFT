import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_help(name: str):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)


def test_hover_entrypoints_expose_help_without_runtime_imports():
    for name in ("run_pybullet_hover_training.py", "run_pybullet_hover_eval.py", "run_pybullet_hover_viz.py", "publish_pybullet_hover_models.py"):
        result = run_help(name)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_training_dry_run_validates_external_contract():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_pybullet_hover_training.py"), "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "profile=ct_att_yawrate_v1" in result.stdout
