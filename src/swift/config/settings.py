from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
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

        raw_required_tasks = mapping.get("required_tasks", ())
        if isinstance(raw_required_tasks, str) or not isinstance(
            raw_required_tasks, Sequence
        ):
            raise ValueError("required_tasks must be a list of task names")
        if any(not isinstance(task, str) or not task.strip() for task in raw_required_tasks):
            raise ValueError("required_tasks must be a list of task names")
        required_tasks = tuple(task for task in raw_required_tasks)
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
