# SWIFT Repository Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `D:\project\SWIFT` into an enterprise-grade Python main repository that validates and orchestrates the existing local PyBullet substrate at `D:\project\pybullet`.

**Architecture:** SWIFT owns configuration, domain contracts, health checks, documentation, and future RL experiment orchestration. `D:\project\pybullet` stays outside the repository and is integrated through a command-level `PyBulletBackend` adapter so the bootstrap remains stable and auditable.

**Tech Stack:** Python 3.11+, setuptools, PyYAML, pytest, PowerShell, Git, local Pixi-backed PyBullet substrate.

---

## File Structure

Create these files in `D:\project\SWIFT`:

```text
D:\project\SWIFT
|-- .gitignore
|-- README.md
|-- pyproject.toml
|-- configs\
|   |-- evaluation.yaml
|   |-- project.yaml
|   |-- simulation.yaml
|   `-- training.yaml
|-- docs\
|   |-- architecture.md
|   |-- implementation-roadmap.md
|   |-- project-charter.md
|   |-- risk-register.md
|   |-- source-materials.md
|   `-- superpowers\
|       |-- plans\
|       `-- specs\
|-- scripts\
|   |-- run_pybullet_smoke.py
|   `-- swift_healthcheck.py
|-- src\
|   `-- swift\
|       |-- __init__.py
|       |-- config\
|       |   |-- __init__.py
|       |   `-- settings.py
|       |-- core\
|       |   |-- __init__.py
|       |   |-- metrics.py
|       |   `-- types.py
|       |-- envs\
|       |   |-- __init__.py
|       |   `-- contracts.py
|       |-- experiments\
|       |   |-- __init__.py
|       |   `-- schema.py
|       |-- rl\
|       |   |-- __init__.py
|       |   |-- apf.py
|       |   |-- hca.py
|       |   `-- ppo.py
|       |-- sim\
|       |   |-- __init__.py
|       |   `-- pybullet_backend.py
|       `-- utils\
|           `-- __init__.py
`-- tests\
    |-- contracts\
    |   `-- test_bootstrap_contracts.py
    |-- config\
    |   `-- test_settings.py
    |-- core\
    |   |-- test_metrics.py
    |   `-- test_types.py
    |-- docs\
    |   `-- test_docs.py
    |-- sim\
    |   `-- test_pybullet_backend.py
    |-- test_package_import.py
    `-- test_scripts.py
```

Keep `D:\project\pybullet` outside SWIFT. Do not copy or edit that repository in this bootstrap.

## Task 1: Package Skeleton And Project Metadata

**Files:**
- Create: `D:\project\SWIFT\tests\test_package_import.py`
- Create: `D:\project\SWIFT\pyproject.toml`
- Create: `D:\project\SWIFT\.gitignore`
- Create: `D:\project\SWIFT\README.md`
- Create: `D:\project\SWIFT\src\swift\__init__.py`
- Create: `D:\project\SWIFT\src\swift\config\__init__.py`
- Create: `D:\project\SWIFT\src\swift\core\__init__.py`
- Create: `D:\project\SWIFT\src\swift\sim\__init__.py`
- Create: `D:\project\SWIFT\src\swift\envs\__init__.py`
- Create: `D:\project\SWIFT\src\swift\rl\__init__.py`
- Create: `D:\project\SWIFT\src\swift\experiments\__init__.py`
- Create: `D:\project\SWIFT\src\swift\utils\__init__.py`

- [ ] **Step 1: Write the failing package import test**

```python
# D:\project\SWIFT\tests\test_package_import.py
import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_exposes_version(self):
        import swift

        self.assertEqual(swift.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m unittest tests.test_package_import -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'swift'`.

- [ ] **Step 3: Create project metadata and package skeleton**

```toml
# D:\project\SWIFT\pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "swift-low-altitude"
version = "0.1.0"
description = "Enterprise main repository for an urban drone delivery 3D obstacle avoidance system."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "PyYAML>=6.0.2,<7",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2,<9",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra"
```

```gitignore
# D:\project\SWIFT\.gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
dist/
build/
*.egg-info/
outputs/
runs/
checkpoints/
*.log
```

````markdown
<!-- D:\project\SWIFT\README.md -->
# SWIFT

SWIFT is the enterprise-grade main repository for the Low-Altitude Swift Wing
urban drone delivery 3D obstacle avoidance project.

The repository orchestrates project configuration, simulation integration,
domain contracts, health checks, documentation, and future PPO/HCA/APF
experiments. The local PyBullet substrate remains at `D:\project\pybullet` and
is accessed through an adapter.

## Quick Start

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts\swift_healthcheck.py
python scripts\run_pybullet_smoke.py --check-only
```

## Repository Boundary

- SWIFT owns enterprise project structure and experiment orchestration.
- `D:\project\pybullet` owns the existing Pixi-managed PyBullet demos.
- The first integration uses command-level health checks instead of importing
  PyBullet internals directly.
````

```python
# D:\project\SWIFT\src\swift\__init__.py
"""SWIFT enterprise main repository package."""

__version__ = "0.1.0"
```

Create these empty package marker files:

```python
# D:\project\SWIFT\src\swift\config\__init__.py
```

```python
# D:\project\SWIFT\src\swift\core\__init__.py
```

```python
# D:\project\SWIFT\src\swift\sim\__init__.py
```

```python
# D:\project\SWIFT\src\swift\envs\__init__.py
```

```python
# D:\project\SWIFT\src\swift\rl\__init__.py
```

```python
# D:\project\SWIFT\src\swift\experiments\__init__.py
```

```python
# D:\project\SWIFT\src\swift\utils\__init__.py
```

- [ ] **Step 4: Install the package in editable dev mode**

Run:

```powershell
python -m pip install -e ".[dev]"
```

Expected: installation succeeds and installs `pytest` plus `PyYAML`.

- [ ] **Step 5: Run package tests**

Run:

```powershell
python -m unittest tests.test_package_import -v
python -m pytest tests\test_package_import.py -q
```

Expected: both commands pass.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore README.md pyproject.toml src\swift tests\test_package_import.py
git commit -m "chore: bootstrap SWIFT python package"
```

## Task 2: Configuration Files And Loader

**Files:**
- Create: `D:\project\SWIFT\tests\config\test_settings.py`
- Create: `D:\project\SWIFT\src\swift\config\settings.py`
- Modify: `D:\project\SWIFT\src\swift\config\__init__.py`
- Create: `D:\project\SWIFT\configs\project.yaml`
- Create: `D:\project\SWIFT\configs\simulation.yaml`
- Create: `D:\project\SWIFT\configs\training.yaml`
- Create: `D:\project\SWIFT\configs\evaluation.yaml`

- [ ] **Step 1: Write failing configuration tests**

```python
# D:\project\SWIFT\tests\config\test_settings.py
from pathlib import Path

import pytest

from swift.config import SimulationSettings, load_simulation_settings, load_yaml_file


def test_load_yaml_file_returns_mapping(tmp_path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text("name: swift\ncount: 2\n", encoding="utf-8")

    assert load_yaml_file(config_path) == {"name": "swift", "count": 2}


def test_load_yaml_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_yaml_file(tmp_path / "missing.yaml")


def test_simulation_settings_resolve_default_paths(tmp_path):
    root = tmp_path / "pybullet"
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"pybullet_root: {root}",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks:",
                "  - test",
                "  - drone-demo",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_simulation_settings(config_path)

    assert settings == SimulationSettings(
        pybullet_root=root,
        pixi_executable=root / ".tools" / "pixi" / "pixi.exe",
        required_tasks=("test", "drone-demo"),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=120,
    )


def test_simulation_settings_require_tasks(tmp_path):
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pybullet_root: D:/project/pybullet",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks: []",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_tasks must contain at least one task"):
        load_simulation_settings(config_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\config\test_settings.py -q
```

Expected: FAIL with import errors for `SimulationSettings`, `load_simulation_settings`, or `load_yaml_file`.

- [ ] **Step 3: Implement configuration loading**

```python
# D:\project\SWIFT\src\swift\config\settings.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return data


@dataclass(frozen=True)
class SimulationSettings:
    pybullet_root: Path
    pixi_executable: Path
    required_tasks: tuple[str, ...]
    check_task: str
    smoke_task: str
    command_timeout_seconds: int

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SimulationSettings":
        pybullet_root = Path(str(mapping["pybullet_root"])).expanduser()
        raw_pixi = Path(str(mapping["pixi_executable"]))
        pixi_executable = raw_pixi if raw_pixi.is_absolute() else pybullet_root / raw_pixi

        required_tasks = tuple(str(task) for task in mapping.get("required_tasks", ()))
        if not required_tasks:
            raise ValueError("required_tasks must contain at least one task")

        timeout = int(mapping.get("command_timeout_seconds", 120))
        if timeout <= 0:
            raise ValueError("command_timeout_seconds must be positive")

        return cls(
            pybullet_root=pybullet_root,
            pixi_executable=pixi_executable,
            required_tasks=required_tasks,
            check_task=str(mapping.get("check_task", "test")),
            smoke_task=str(mapping.get("smoke_task", "drone-demo")),
            command_timeout_seconds=timeout,
        )


def load_simulation_settings(path: str | Path) -> SimulationSettings:
    return SimulationSettings.from_mapping(load_yaml_file(path))
```

```python
# D:\project\SWIFT\src\swift\config\__init__.py
from swift.config.settings import (
    SimulationSettings,
    load_simulation_settings,
    load_yaml_file,
)

__all__ = [
    "SimulationSettings",
    "load_simulation_settings",
    "load_yaml_file",
]
```

- [ ] **Step 4: Add repository configuration files**

```yaml
# D:\project\SWIFT\configs\project.yaml
project:
  name: SWIFT
  title: Low-Altitude Swift Wing
  domain: urban drone delivery 3D obstacle avoidance
  schedule:
    start: 2026-05
    finish: 2027-05
  technical_line:
    decision: PPO
    perception: HCA
    safety_prior: APF
    simulation: PyBullet
```

```yaml
# D:\project\SWIFT\configs\simulation.yaml
pybullet_root: D:\project\pybullet
pixi_executable: .tools\pixi\pixi.exe
required_tasks:
  - test
  - drone-demo
check_task: test
smoke_task: drone-demo
command_timeout_seconds: 120
```

```yaml
# D:\project\SWIFT\configs\training.yaml
ppo:
  rollout_steps: 2048
  minibatch_size: 64
  update_epochs: 10
  clip_range: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coef: 0.01
  value_loss_coef: 0.5
  max_grad_norm: 0.5
baseline:
  model: MLP
  purpose: stable demonstrable baseline before HCA and APF fusion
```

```yaml
# D:\project\SWIFT\configs\evaluation.yaml
metrics:
  - success_rate
  - collision_rate
  - average_path_length
  - path_smoothness
  - convergence_episodes
  - minimum_inter_drone_distance
artifacts:
  root: outputs
  episode_logs: outputs\episodes
  experiment_reports: outputs\reports
```

- [ ] **Step 5: Run configuration tests**

Run:

```powershell
python -m pytest tests\config\test_settings.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add configs src\swift\config tests\config
git commit -m "feat: add SWIFT configuration loader"
```

## Task 3: PyBullet Backend Adapter

**Files:**
- Create: `D:\project\SWIFT\tests\sim\test_pybullet_backend.py`
- Create: `D:\project\SWIFT\src\swift\sim\pybullet_backend.py`
- Modify: `D:\project\SWIFT\src\swift\sim\__init__.py`

- [ ] **Step 1: Write failing backend tests**

```python
# D:\project\SWIFT\tests\sim\test_pybullet_backend.py
from pathlib import Path

from swift.config import SimulationSettings
from swift.sim import PyBulletBackend


def make_settings(root: Path) -> SimulationSettings:
    return SimulationSettings(
        pybullet_root=root,
        pixi_executable=root / ".tools" / "pixi" / "pixi.exe",
        required_tasks=("test", "drone-demo"),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )


def make_fake_pybullet_root(root: Path) -> None:
    pixi = root / ".tools" / "pixi" / "pixi.exe"
    pixi.parent.mkdir(parents=True)
    pixi.write_text("fake pixi executable", encoding="utf-8")
    (root / "pixi.toml").write_text(
        "\n".join(
            [
                "[tasks]",
                'test = "python -m unittest discover tests"',
                'drone-demo = "python scripts/run_drone_llm.py"',
            ]
        ),
        encoding="utf-8",
    )


def test_inspect_reports_ok_for_valid_substrate(tmp_path):
    make_fake_pybullet_root(tmp_path)

    health = PyBulletBackend(make_settings(tmp_path)).inspect()

    assert health.ok is True
    assert [check.name for check in health.checks] == [
        "pybullet_root",
        "pixi_toml",
        "pixi_executable",
        "required_task:test",
        "required_task:drone-demo",
    ]


def test_inspect_reports_all_missing_parts(tmp_path):
    missing_root = tmp_path / "missing"

    health = PyBulletBackend(make_settings(missing_root)).inspect()

    assert health.ok is False
    assert any(check.name == "pybullet_root" and not check.ok for check in health.checks)
    assert any(check.name == "pixi_toml" and not check.ok for check in health.checks)
    assert any(check.name == "pixi_executable" and not check.ok for check in health.checks)


def test_build_task_command_uses_configured_pixi_path(tmp_path):
    make_fake_pybullet_root(tmp_path)
    backend = PyBulletBackend(make_settings(tmp_path))

    command = backend.build_task_command("test")

    assert command == [str(tmp_path / ".tools" / "pixi" / "pixi.exe"), "run", "test"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\sim\test_pybullet_backend.py -q
```

Expected: FAIL with import errors for `PyBulletBackend`.

- [ ] **Step 3: Implement backend inspection and command construction**

```python
# D:\project\SWIFT\src\swift\sim\pybullet_backend.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from swift.config import SimulationSettings


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class BackendHealth:
    ok: bool
    checks: tuple[CheckResult, ...]

    def summary_lines(self) -> list[str]:
        lines = []
        for check in self.checks:
            status = "OK" if check.ok else "FAIL"
            lines.append(f"[{status}] {check.name}: {check.detail}")
        return lines


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class PyBulletBackend:
    def __init__(self, settings: SimulationSettings):
        self.settings = settings

    def inspect(self) -> BackendHealth:
        checks: list[CheckResult] = []
        root = self.settings.pybullet_root
        pixi_toml = root / "pixi.toml"

        checks.append(
            CheckResult(
                name="pybullet_root",
                ok=root.exists() and root.is_dir(),
                detail=str(root),
            )
        )
        checks.append(
            CheckResult(
                name="pixi_toml",
                ok=pixi_toml.exists() and pixi_toml.is_file(),
                detail=str(pixi_toml),
            )
        )
        checks.append(
            CheckResult(
                name="pixi_executable",
                ok=self.settings.pixi_executable.exists()
                and self.settings.pixi_executable.is_file(),
                detail=str(self.settings.pixi_executable),
            )
        )

        pixi_text = pixi_toml.read_text(encoding="utf-8") if pixi_toml.exists() else ""
        for task in self.settings.required_tasks:
            checks.append(
                CheckResult(
                    name=f"required_task:{task}",
                    ok=f"{task} =" in pixi_text,
                    detail=f"task '{task}' declared in {pixi_toml}",
                )
            )

        return BackendHealth(
            ok=all(check.ok for check in checks),
            checks=tuple(checks),
        )

    def build_task_command(self, task: str) -> list[str]:
        return [str(self.settings.pixi_executable), "run", task]

    def run_task(self, task: str, timeout_seconds: int | None = None) -> CommandResult:
        command = self.build_task_command(task)
        completed = subprocess.run(
            command,
            cwd=self.settings.pybullet_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds or self.settings.command_timeout_seconds,
            check=False,
        )
        return CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
```

```python
# D:\project\SWIFT\src\swift\sim\__init__.py
from swift.sim.pybullet_backend import (
    BackendHealth,
    CheckResult,
    CommandResult,
    PyBulletBackend,
)

__all__ = [
    "BackendHealth",
    "CheckResult",
    "CommandResult",
    "PyBulletBackend",
]
```

- [ ] **Step 4: Run backend tests**

Run:

```powershell
python -m pytest tests\sim\test_pybullet_backend.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src\swift\sim tests\sim
git commit -m "feat: add PyBullet substrate adapter"
```

## Task 4: Health Check And Smoke CLI Scripts

**Files:**
- Create: `D:\project\SWIFT\tests\test_scripts.py`
- Create: `D:\project\SWIFT\scripts\swift_healthcheck.py`
- Create: `D:\project\SWIFT\scripts\run_pybullet_smoke.py`

- [ ] **Step 1: Write failing CLI tests**

```python
# D:\project\SWIFT\tests\test_scripts.py
import subprocess
import sys
from pathlib import Path


def make_fake_config(tmp_path: Path) -> Path:
    root = tmp_path / "pybullet"
    pixi = root / ".tools" / "pixi" / "pixi.exe"
    pixi.parent.mkdir(parents=True)
    pixi.write_text("fake pixi executable", encoding="utf-8")
    (root / "pixi.toml").write_text(
        "\n".join(
            [
                "[tasks]",
                'test = "python -m unittest discover tests"',
                'drone-demo = "python scripts/run_drone_llm.py"',
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "simulation.yaml"
    config.write_text(
        "\n".join(
            [
                f"pybullet_root: {root}",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks:",
                "  - test",
                "  - drone-demo",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 5",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_healthcheck_check_only_passes_with_fake_substrate(tmp_path):
    config = make_fake_config(tmp_path)

    result = subprocess.run(
        [sys.executable, "scripts/swift_healthcheck.py", "--config", str(config)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "SWIFT pybullet substrate: OK" in result.stdout


def test_smoke_check_only_does_not_execute_pixi(tmp_path):
    config = make_fake_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_smoke.py",
            "--config",
            str(config),
            "--check-only",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Full smoke command:" in result.stdout
    assert "pixi.exe run drone-demo" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_scripts.py -q
```

Expected: FAIL because `scripts\swift_healthcheck.py` and `scripts\run_pybullet_smoke.py` do not exist.

- [ ] **Step 3: Implement the health check script**

```python
# D:\project\SWIFT\scripts\swift_healthcheck.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings
from swift.sim import PyBulletBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check SWIFT local substrate readiness.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "simulation.yaml"),
        help="Path to simulation YAML configuration.",
    )
    args = parser.parse_args(argv)

    settings = load_simulation_settings(args.config)
    backend = PyBulletBackend(settings)
    health = backend.inspect()

    print(f"SWIFT pybullet substrate: {'OK' if health.ok else 'FAIL'}")
    for line in health.summary_lines():
        print(line)

    print("Full smoke command:")
    print(" ".join(backend.build_task_command(settings.smoke_task)))
    return 0 if health.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement the smoke script**

```python
# D:\project\SWIFT\scripts\run_pybullet_smoke.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings
from swift.sim import PyBulletBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or inspect the PyBullet smoke command.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "simulation.yaml"),
        help="Path to simulation YAML configuration.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate paths and print the command.",
    )
    args = parser.parse_args(argv)

    settings = load_simulation_settings(args.config)
    backend = PyBulletBackend(settings)
    health = backend.inspect()
    if not health.ok:
        print("SWIFT pybullet substrate: FAIL")
        for line in health.summary_lines():
            print(line)
        return 1

    command = backend.build_task_command(settings.smoke_task)
    print("Full smoke command:")
    print(" ".join(command))

    if args.check_only:
        return 0

    result = backend.run_task(settings.smoke_task)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run CLI tests**

Run:

```powershell
python -m pytest tests\test_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run scripts against the real configured substrate**

Run:

```powershell
python scripts\swift_healthcheck.py
python scripts\run_pybullet_smoke.py --check-only
```

Expected: both commands exit `0`, print `SWIFT pybullet substrate: OK`, and show the full `pixi.exe run drone-demo` command.

- [ ] **Step 7: Commit**

```powershell
git add scripts tests\test_scripts.py
git commit -m "feat: add SWIFT health check scripts"
```

## Task 5: Core Domain Types And Metrics

**Files:**
- Create: `D:\project\SWIFT\tests\core\test_types.py`
- Create: `D:\project\SWIFT\tests\core\test_metrics.py`
- Create: `D:\project\SWIFT\src\swift\core\types.py`
- Create: `D:\project\SWIFT\src\swift\core\metrics.py`
- Modify: `D:\project\SWIFT\src\swift\core\__init__.py`

- [ ] **Step 1: Write failing domain type tests**

```python
# D:\project\SWIFT\tests\core\test_types.py
import pytest

from swift.core import DroneAction, DroneState, EpisodeMetrics, ObstacleState, RewardBreakdown


def test_drone_state_is_three_dimensional():
    state = DroneState(position=(1.0, 2.0, 3.0), velocity=(0.1, 0.2, 0.3), yaw=0.5)

    assert state.position == (1.0, 2.0, 3.0)
    assert state.velocity == (0.1, 0.2, 0.3)


def test_drone_state_rejects_invalid_vector_length():
    with pytest.raises(ValueError, match="position must contain exactly 3 values"):
        DroneState(position=(1.0, 2.0), velocity=(0.0, 0.0, 0.0), yaw=0.0)


def test_reward_breakdown_total_sums_terms():
    reward = RewardBreakdown(
        arrive=10.0,
        approach=1.5,
        obstacle=-2.0,
        smoothness=-0.5,
        timeliness=-1.0,
    )

    assert reward.total == 8.0


def test_episode_metrics_success_flag_is_explicit():
    metrics = EpisodeMetrics(
        reached_goal=True,
        collided=False,
        timed_out=False,
        path_length=12.0,
        path_smoothness=0.2,
        minimum_safety_distance=1.5,
        steps=120,
    )

    assert metrics.success is True
```

- [ ] **Step 2: Write failing metric tests**

```python
# D:\project\SWIFT\tests\core\test_metrics.py
from swift.core import (
    compute_minimum_distance,
    compute_path_length,
    compute_path_smoothness,
)


def test_compute_path_length_sums_segments():
    points = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 12.0)]

    assert compute_path_length(points) == 17.0


def test_compute_path_smoothness_is_zero_for_straight_path():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

    assert compute_path_smoothness(points) == 0.0


def test_compute_minimum_distance_between_two_tracks():
    ownship = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    intruder = [(0.0, 3.0, 0.0), (2.0, 4.0, 0.0)]

    assert compute_minimum_distance(ownship, intruder) == 3.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\core -q
```

Expected: FAIL with import errors for the core types and functions.

- [ ] **Step 4: Implement domain types**

```python
# D:\project\SWIFT\src\swift\core\types.py
from __future__ import annotations

from dataclasses import dataclass

Vector3 = tuple[float, float, float]


def _validate_vector3(name: str, value: tuple[float, ...]) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class DroneState:
    position: Vector3
    velocity: Vector3
    yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _validate_vector3("position", self.position))
        object.__setattr__(self, "velocity", _validate_vector3("velocity", self.velocity))
        object.__setattr__(self, "yaw", float(self.yaw))


@dataclass(frozen=True)
class DroneAction:
    speed: float
    heading_delta: float
    climb_rate: float


@dataclass(frozen=True)
class ObstacleState:
    position: Vector3
    radius: float
    velocity: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _validate_vector3("position", self.position))
        object.__setattr__(self, "velocity", _validate_vector3("velocity", self.velocity))
        if self.radius <= 0:
            raise ValueError("radius must be positive")


@dataclass(frozen=True)
class RewardBreakdown:
    arrive: float
    approach: float
    obstacle: float
    smoothness: float
    timeliness: float

    @property
    def total(self) -> float:
        return self.arrive + self.approach + self.obstacle + self.smoothness + self.timeliness


@dataclass(frozen=True)
class EpisodeMetrics:
    reached_goal: bool
    collided: bool
    timed_out: bool
    path_length: float
    path_smoothness: float
    minimum_safety_distance: float
    steps: int

    @property
    def success(self) -> bool:
        return self.reached_goal and not self.collided and not self.timed_out
```

- [ ] **Step 5: Implement metric functions**

```python
# D:\project\SWIFT\src\swift\core\metrics.py
from __future__ import annotations

import math

from swift.core.types import Vector3


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def compute_path_length(points: list[Vector3] | tuple[Vector3, ...]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_distance(start, end) for start, end in zip(points, points[1:]))


def compute_path_smoothness(points: list[Vector3] | tuple[Vector3, ...]) -> float:
    if len(points) < 3:
        return 0.0

    total_turn = 0.0
    for previous, current, following in zip(points, points[1:], points[2:]):
        v1 = tuple(current[i] - previous[i] for i in range(3))
        v2 = tuple(following[i] - current[i] for i in range(3))
        n1 = math.sqrt(sum(value * value for value in v1))
        n2 = math.sqrt(sum(value * value for value in v2))
        if n1 == 0.0 or n2 == 0.0:
            continue
        cos_angle = sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        total_turn += math.acos(cos_angle)
    return total_turn


def compute_minimum_distance(
    ownship: list[Vector3] | tuple[Vector3, ...],
    intruder: list[Vector3] | tuple[Vector3, ...],
) -> float:
    if not ownship or not intruder:
        raise ValueError("ownship and intruder tracks must not be empty")
    return min(_distance(a, b) for a in ownship for b in intruder)
```

```python
# D:\project\SWIFT\src\swift\core\__init__.py
from swift.core.metrics import (
    compute_minimum_distance,
    compute_path_length,
    compute_path_smoothness,
)
from swift.core.types import (
    DroneAction,
    DroneState,
    EpisodeMetrics,
    ObstacleState,
    RewardBreakdown,
    Vector3,
)

__all__ = [
    "DroneAction",
    "DroneState",
    "EpisodeMetrics",
    "ObstacleState",
    "RewardBreakdown",
    "Vector3",
    "compute_minimum_distance",
    "compute_path_length",
    "compute_path_smoothness",
]
```

- [ ] **Step 6: Run core tests**

Run:

```powershell
python -m pytest tests\core -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src\swift\core tests\core
git commit -m "feat: add drone domain contracts and metrics"
```

## Task 6: Environment, RL, And Experiment Bootstrap Contracts

**Files:**
- Create: `D:\project\SWIFT\tests\contracts\test_bootstrap_contracts.py`
- Create: `D:\project\SWIFT\src\swift\envs\contracts.py`
- Modify: `D:\project\SWIFT\src\swift\envs\__init__.py`
- Create: `D:\project\SWIFT\src\swift\rl\ppo.py`
- Create: `D:\project\SWIFT\src\swift\rl\hca.py`
- Create: `D:\project\SWIFT\src\swift\rl\apf.py`
- Modify: `D:\project\SWIFT\src\swift\rl\__init__.py`
- Create: `D:\project\SWIFT\src\swift\experiments\schema.py`
- Modify: `D:\project\SWIFT\src\swift\experiments\__init__.py`

- [ ] **Step 1: Write failing contract tests**

```python
# D:\project\SWIFT\tests\contracts\test_bootstrap_contracts.py
import pytest

from swift.envs import BootstrapDroneEnv, UnsupportedOperationError
from swift.experiments import ExperimentMetric, ExperimentSpec
from swift.rl import APFConfig, HCAConfig, PPOConfig


def test_bootstrap_environment_declares_shapes_and_blocks_runtime_use():
    env = BootstrapDroneEnv(observation_size=24, action_size=3)

    assert env.observation_shape == (24,)
    assert env.action_shape == (3,)
    with pytest.raises(UnsupportedOperationError, match="outside bootstrap scope"):
        env.reset()


def test_rl_configs_capture_project_defaults():
    assert PPOConfig().clip_range == 0.2
    assert HCAConfig().target_attention_heads == 4
    assert APFConfig().repulsive_gain == 1.0


def test_experiment_spec_names_ablation_variants():
    spec = ExperimentSpec(
        name="bootstrap-ablation-matrix",
        variants=("mlp_ppo", "hca_ppo", "hca_apf_ppo"),
        metrics=(
            ExperimentMetric.SUCCESS_RATE,
            ExperimentMetric.PATH_SMOOTHNESS,
        ),
    )

    assert spec.variants == ("mlp_ppo", "hca_ppo", "hca_apf_ppo")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\contracts\test_bootstrap_contracts.py -q
```

Expected: FAIL with import errors for environment, RL, and experiment contracts.

- [ ] **Step 3: Implement environment contract**

```python
# D:\project\SWIFT\src\swift\envs\contracts.py
from __future__ import annotations

from dataclasses import dataclass


class UnsupportedOperationError(RuntimeError):
    """Raised when a bootstrap contract is called as a runtime implementation."""


@dataclass(frozen=True)
class BootstrapDroneEnv:
    observation_size: int
    action_size: int

    @property
    def observation_shape(self) -> tuple[int]:
        return (self.observation_size,)

    @property
    def action_shape(self) -> tuple[int]:
        return (self.action_size,)

    def reset(self) -> None:
        raise UnsupportedOperationError("Full RL environment runtime is outside bootstrap scope")

    def step(self, action: object) -> None:
        raise UnsupportedOperationError("Full RL environment runtime is outside bootstrap scope")
```

```python
# D:\project\SWIFT\src\swift\envs\__init__.py
from swift.envs.contracts import BootstrapDroneEnv, UnsupportedOperationError

__all__ = ["BootstrapDroneEnv", "UnsupportedOperationError"]
```

- [ ] **Step 4: Implement RL bootstrap configs**

```python
# D:\project\SWIFT\src\swift\rl\ppo.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 2048
    minibatch_size: int = 64
    update_epochs: int = 10
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
```

```python
# D:\project\SWIFT\src\swift\rl\hca.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HCAConfig:
    target_attention_heads: int = 4
    threat_attention_heads: int = 4
    embedding_dim: int = 128
    dropout: float = 0.1
```

```python
# D:\project\SWIFT\src\swift\rl\apf.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APFConfig:
    attractive_gain: float = 1.0
    repulsive_gain: float = 1.0
    influence_radius: float = 2.0
    epsilon: float = 1e-6
```

```python
# D:\project\SWIFT\src\swift\rl\__init__.py
from swift.rl.apf import APFConfig
from swift.rl.hca import HCAConfig
from swift.rl.ppo import PPOConfig

__all__ = ["APFConfig", "HCAConfig", "PPOConfig"]
```

- [ ] **Step 5: Implement experiment schema**

```python
# D:\project\SWIFT\src\swift\experiments\schema.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExperimentMetric(StrEnum):
    SUCCESS_RATE = "success_rate"
    COLLISION_RATE = "collision_rate"
    AVERAGE_PATH_LENGTH = "average_path_length"
    PATH_SMOOTHNESS = "path_smoothness"
    CONVERGENCE_EPISODES = "convergence_episodes"
    MINIMUM_INTER_DRONE_DISTANCE = "minimum_inter_drone_distance"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    variants: tuple[str, ...]
    metrics: tuple[ExperimentMetric, ...]
```

```python
# D:\project\SWIFT\src\swift\experiments\__init__.py
from swift.experiments.schema import ExperimentMetric, ExperimentSpec

__all__ = ["ExperimentMetric", "ExperimentSpec"]
```

- [ ] **Step 6: Run contract tests**

Run:

```powershell
python -m pytest tests\contracts\test_bootstrap_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src\swift\envs src\swift\rl src\swift\experiments tests\contracts
git commit -m "feat: add bootstrap environment and experiment contracts"
```

## Task 7: Enterprise Documentation Assets

**Files:**
- Create: `D:\project\SWIFT\tests\docs\test_docs.py`
- Create: `D:\project\SWIFT\docs\project-charter.md`
- Create: `D:\project\SWIFT\docs\architecture.md`
- Create: `D:\project\SWIFT\docs\implementation-roadmap.md`
- Create: `D:\project\SWIFT\docs\risk-register.md`
- Create: `D:\project\SWIFT\docs\source-materials.md`

- [ ] **Step 1: Write failing documentation tests**

```python
# D:\project\SWIFT\tests\docs\test_docs.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_required_enterprise_docs_exist():
    required = [
        "docs/project-charter.md",
        "docs/architecture.md",
        "docs/implementation-roadmap.md",
        "docs/risk-register.md",
        "docs/source-materials.md",
    ]

    for relative in required:
        assert (ROOT / relative).exists(), f"Missing {relative}"


def test_project_charter_records_core_technical_line():
    text = (ROOT / "docs" / "project-charter.md").read_text(encoding="utf-8")

    assert "PPO" in text
    assert "HCA" in text
    assert "APF" in text
    assert "PyBullet" in text


def test_risk_register_covers_bootstrap_risks():
    text = (ROOT / "docs" / "risk-register.md").read_text(encoding="utf-8")

    for risk in ["Sparse reward", "PPO-HCA coupling", "APF local minima", "GPU availability"]:
        assert risk in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests\docs\test_docs.py -q
```

Expected: FAIL because required docs do not exist.

- [ ] **Step 3: Create project charter**

```markdown
<!-- D:\project\SWIFT\docs\project-charter.md -->
# Project Charter

## Mission

SWIFT builds an enterprise-grade main repository for an urban drone delivery
3D obstacle avoidance system. The system target is a demonstrable, verifiable,
and extensible simulation platform for safe low-altitude delivery navigation.

## Technical Line

- PPO is the decision baseline for continuous action control.
- HCA is the perception upgrade for layered target and threat attention.
- APF is the safety prior that provides interpretable attraction and repulsion
  vectors.
- PyBullet is the simulation substrate for 3D urban low-altitude validation.

## Bootstrap Boundary

The bootstrap creates repository structure, configuration, documentation,
health checks, and contracts. Full PPO training, HCA model execution, APF-HCA
fusion, and long-running GPU experiments are follow-on stages.

## Local Substrate

The local PyBullet substrate is `D:\project\pybullet`. SWIFT validates it
through a command-level adapter and does not copy it into this repository.
```

- [ ] **Step 4: Create architecture document**

```markdown
<!-- D:\project\SWIFT\docs\architecture.md -->
# Architecture

## System Boundary

SWIFT is the main project repository. `D:\project\pybullet` is an external local
simulation substrate.

## Layers

1. Configuration layer: YAML files define project, simulation, training, and
   evaluation settings.
2. Simulation adapter layer: `PyBulletBackend` validates the local substrate and
   constructs Pixi task commands.
3. Core domain layer: drone states, actions, obstacles, rewards, and metrics.
4. Environment contract layer: Gymnasium-style boundaries for later runtime
   implementation.
5. RL bootstrap layer: PPO, HCA, and APF configuration contracts.
6. Experiment layer: ablation variants and metric naming.

## Integration Rule

The first integration uses command boundaries. Direct Python imports from the
PyBullet substrate are deferred until SWIFT owns a stable environment API.
```

- [ ] **Step 5: Create roadmap document**

```markdown
<!-- D:\project\SWIFT\docs\implementation-roadmap.md -->
# Implementation Roadmap

## Stage 0: Repository Bootstrap

Create package metadata, configuration, source package structure, health checks,
domain contracts, documentation, and tests.

## Stage 1: PPO+MLP Baseline

Build the baseline environment loop and PPO+MLP policy path. This is the safe
demonstrable baseline.

## Stage 2: PPO+HCA Upgrade

Replace flat MLP perception with target and threat attention layers. Preserve
the PPO baseline as the comparison target.

## Stage 3: HCA+APF Fusion

Embed APF attraction and repulsion vectors into the HCA perception flow. Measure
convergence speed, success rate, path smoothness, and minimum safety distance.

## Stage 4: Multi-Scenario Evaluation

Run ablations across building density, dynamic drone count, delivery targets,
and altitude preferences. Produce reports, demo video inputs, and final project
evidence.
```

- [ ] **Step 6: Create risk register**

```markdown
<!-- D:\project\SWIFT\docs\risk-register.md -->
# Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Sparse reward | Training may fail to discover useful behavior early. | Keep PPO+MLP baseline small, add shaped approach and safety terms, track convergence episodes. |
| PPO-HCA coupling | Attention layers may destabilize PPO updates. | Use dimension checks, normalization, gradient clipping, and compare against the MLP baseline. |
| APF local minima | APF may bias the policy toward dead ends in U-shaped or dense obstacles. | Keep HCA-only ablation, measure APF contribution, and let RL override APF features. |
| GPU availability | Long training may be blocked by local compute limits. | Keep CPU smoke tests, small deterministic scenarios, and prepare cloud GPU configuration later. |
| PyBullet substrate drift | `D:\project\pybullet` may change independently of SWIFT. | Validate through health checks and record exact adapter assumptions. |
| Demo reproducibility | GUI demos and videos can diverge from testable evidence. | Treat headless smoke tests and logged metrics as the source of truth. |
```

- [ ] **Step 7: Create source-materials document**

```markdown
<!-- D:\project\SWIFT\docs\source-materials.md -->
# Source Materials

## User-Provided Documents

- Project presentation: urban drone delivery 3D obstacle avoidance PPTX from
  the Telegram Desktop downloads directory.
- Preparation document: DOCX covering prior work, technical route, schedule,
  resource readiness, and team roles.
- Innovation training application: PDF covering formal project fields, research
  rationale, implementation plan, budget, and expected outcomes.

## Extracted Claims

- Project period: May 2026 to May 2027.
- Technical route: PPO decision base, HCA layered perception, APF safety prior,
  PyBullet simulation validation.
- Development strategy: baseline first, HCA upgrade second, APF fusion third,
  multi-scenario evaluation last.
- Expected outcomes: simulation system, research report, demo evidence, and
  possible software copyright application.

## Local Code Substrate

`D:\project\pybullet` contains a Pixi-managed PyBullet environment, tests,
`gym-pybullet-drones`, and a runnable headless drone demo. SWIFT integrates it
as an external local dependency.
```

- [ ] **Step 8: Run documentation tests**

Run:

```powershell
python -m pytest tests\docs\test_docs.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```powershell
git add docs\project-charter.md docs\architecture.md docs\implementation-roadmap.md docs\risk-register.md docs\source-materials.md tests\docs
git commit -m "docs: add enterprise project documentation"
```

## Task 8: Full Bootstrap Verification

**Files:**
- Modify only if a verification failure exposes a real defect in files created by Tasks 1-7.

- [ ] **Step 1: Run the full test suite**

Run:

```powershell
python -m pytest
```

Expected: all tests pass.

- [ ] **Step 2: Run SWIFT health check against real pybullet substrate**

Run:

```powershell
python scripts\swift_healthcheck.py
```

Expected: exit code `0`, with checks passing for:

- `pybullet_root`
- `pixi_toml`
- `pixi_executable`
- `required_task:test`
- `required_task:drone-demo`

- [ ] **Step 3: Run check-only smoke command**

Run:

```powershell
python scripts\run_pybullet_smoke.py --check-only
```

Expected: exit code `0` and printed command ending in `.tools\pixi\pixi.exe run drone-demo`.

- [ ] **Step 4: Run optional live substrate test**

Run:

```powershell
python scripts\run_pybullet_smoke.py
```

Expected: exit code `0` and the existing PyBullet drone demo output. If this fails while `--check-only` passes, record the stderr and inspect the existing `D:\project\pybullet` demo before changing SWIFT.

- [ ] **Step 5: Inspect Git status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on `master`.

- [ ] **Step 6: Final commit if verification required small fixes**

Run only if Step 1-4 required code changes:

```powershell
git add .
git commit -m "fix: stabilize bootstrap verification"
```

Expected: commit succeeds and `git status --short --branch` is clean.

## Self-Review Checklist

- Spec coverage: this plan covers repository metadata, configuration, PyBullet adapter, scripts, domain contracts, environment/RL/experiment boundaries, documentation, and verification gates.
- Placeholders: the plan contains no placeholder sections or deferred implementation slots inside the bootstrap scope.
- Type consistency: `SimulationSettings`, `PyBulletBackend`, `DroneState`, `EpisodeMetrics`, `BootstrapDroneEnv`, `PPOConfig`, `HCAConfig`, `APFConfig`, and `ExperimentSpec` are introduced before later tasks reference them.
- Risk control: full PPO, HCA runtime, APF fusion, GUI demos, and GPU training remain outside bootstrap scope as required by the approved design.
