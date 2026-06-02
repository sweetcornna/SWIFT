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
    assert "SWIFT pybullet substrate: OK" in result.stdout
    assert "Full smoke command:" in result.stdout
    assert "pixi.exe run drone-demo" in result.stdout
