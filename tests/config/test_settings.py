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


def test_simulation_settings_rejects_scalar_required_tasks(tmp_path):
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pybullet_root: D:/project/pybullet",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks: test",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_tasks must be a list of task names"):
        load_simulation_settings(config_path)


def test_simulation_settings_rejects_non_string_task_names(tmp_path):
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pybullet_root: D:/project/pybullet",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks:",
                "  - test",
                "  - 42",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_tasks must be a list of task names"):
        load_simulation_settings(config_path)
