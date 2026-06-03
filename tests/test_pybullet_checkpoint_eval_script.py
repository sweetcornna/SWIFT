import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pybullet_checkpoint_eval_script_dry_run_validates_configs(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_checkpoint_eval.py",
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT PyBullet checkpoint evaluation config OK" in result.stdout


def test_pybullet_checkpoint_eval_script_dry_run_avoids_runtime_imports(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"placeholder")
    code = f"""
import builtins

blocked = {{
    'swift.experiments.pybullet_checkpoint_evaluator',
    'swift.sim',
    'torch',
    'pybullet',
    'gym_pybullet_drones',
}}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked or any(name.startswith(prefix + '.') for prefix in blocked):
        raise RuntimeError(f'blocked import: {{name}}')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from scripts.run_pybullet_checkpoint_eval import main
raise SystemExit(main(['--checkpoint', r'{checkpoint}', '--dry-run']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT PyBullet checkpoint evaluation config OK" in result.stdout


def test_pybullet_checkpoint_eval_script_dry_run_rejects_invalid_evaluator_args(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"placeholder")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_checkpoint_eval.py",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "episodes must be positive" in result.stderr


def test_pybullet_checkpoint_eval_script_dry_run_rejects_missing_checkpoint(tmp_path: Path):
    missing_checkpoint = tmp_path / "missing.ckpt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_checkpoint_eval.py",
            "--checkpoint",
            str(missing_checkpoint),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "checkpoint must exist" in result.stderr
