from pathlib import Path
from unittest.mock import patch

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


def test_inspect_handles_pixi_toml_directory_without_crashing(tmp_path):
    pixi = tmp_path / ".tools" / "pixi" / "pixi.exe"
    pixi.parent.mkdir(parents=True)
    pixi.write_text("fake pixi executable", encoding="utf-8")
    (tmp_path / "pixi.toml").mkdir()

    health = PyBulletBackend(make_settings(tmp_path)).inspect()

    assert health.ok is False
    assert any(check.name == "pixi_toml" and not check.ok for check in health.checks)
    assert any(
        check.name == "required_task:test" and not check.ok for check in health.checks
    )


def test_inspect_parses_exact_toml_task_keys(tmp_path):
    pixi = tmp_path / ".tools" / "pixi" / "pixi.exe"
    pixi.parent.mkdir(parents=True)
    pixi.write_text("fake pixi executable", encoding="utf-8")
    (tmp_path / "pixi.toml").write_text(
        "\n".join(
            [
                "[tasks]",
                'unit-test = "python -m unittest"',
                'drone-demo="python scripts/run_drone_llm.py"',
            ]
        ),
        encoding="utf-8",
    )

    health = PyBulletBackend(make_settings(tmp_path)).inspect()

    assert health.ok is False
    assert any(
        check.name == "required_task:test" and not check.ok for check in health.checks
    )
    assert any(
        check.name == "required_task:drone-demo" and check.ok for check in health.checks
    )


def test_run_task_returns_command_result_on_timeout(tmp_path):
    make_fake_pybullet_root(tmp_path)
    backend = PyBulletBackend(make_settings(tmp_path))

    import subprocess

    with patch("swift.sim.pybullet_backend.subprocess.run") as run:
        run.side_effect = subprocess.TimeoutExpired(
            cmd=backend.build_task_command("drone-demo"),
            timeout=1,
            output="partial stdout",
            stderr="partial stderr",
        )

        result = backend.run_task("drone-demo", timeout_seconds=1)

    assert result.returncode == 124
    assert "partial stdout" in result.stdout
    assert "Timed out after 1 seconds" in result.stderr
    assert "partial stderr" in result.stderr
