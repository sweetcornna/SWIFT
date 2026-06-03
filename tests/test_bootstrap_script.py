import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_quick_dry_run_plans_dev_install_and_core_verification(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap.py",
            "--mode",
            "quick",
            "--venv-path",
            str(tmp_path / "venv"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT bootstrap mode=quick dry_run=True" in result.stdout
    assert "pip install -e .[dev]" in result.stdout
    assert "-m pytest" in result.stdout
    assert "scripts\\swift_healthcheck.py" in result.stdout or "scripts/swift_healthcheck.py" in result.stdout


def test_bootstrap_train_dry_run_plans_train_extra_and_ppo_smoke(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap.py",
            "--mode",
            "train",
            "--venv-path",
            str(tmp_path / "venv"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pip install -e .[dev,train]" in result.stdout
    assert "scripts\\run_ppo_mlp_smoke.py" in result.stdout or "scripts/run_ppo_mlp_smoke.py" in result.stdout


def test_bootstrap_pybullet_dry_run_uses_pixi_python_and_runtime_smoke(tmp_path: Path):
    pybullet_root = tmp_path / "pybullet"
    pixi_python = pybullet_root / ".pixi" / "envs" / "default" / "python.exe"
    pixi_python.parent.mkdir(parents=True)
    pixi_python.write_text("fake python", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap.py",
            "--mode",
            "pybullet",
            "--pybullet-root",
            str(pybullet_root),
            "--pybullet-venv",
            str(tmp_path / "pybullet-venv"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(pixi_python) in result.stdout
    assert "--system-site-packages" in result.stdout
    assert "scripts\\run_pybullet_smoke.py" in result.stdout or "scripts/run_pybullet_smoke.py" in result.stdout
    assert "scripts\\run_pybullet_runtime_smoke.py" in result.stdout or "scripts/run_pybullet_runtime_smoke.py" in result.stdout


def test_bootstrap_powershell_wrapper_dry_run_delegates_to_python_entrypoint(tmp_path: Path):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        return

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/bootstrap.ps1",
            "-Mode",
            "quick",
            "-VenvPath",
            str(tmp_path / "venv"),
            "-DryRun",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "SWIFT bootstrap mode=quick dry_run=True" in result.stdout
