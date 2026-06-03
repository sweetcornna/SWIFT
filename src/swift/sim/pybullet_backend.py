from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass
from typing import Any

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

        task_table = self._load_task_table(pixi_toml)
        for task in self.settings.required_tasks:
            checks.append(
                CheckResult(
                    name=f"required_task:{task}",
                    ok=task in task_table,
                    detail=f"task '{task}' declared in {pixi_toml}",
                )
            )

        return BackendHealth(
            ok=all(check.ok for check in checks),
            checks=tuple(checks),
        )

    def build_task_command(self, task: str) -> list[str]:
        return [str(self.settings.pixi_executable), "run", task]

    def _load_task_table(self, pixi_toml: Any) -> dict[str, Any]:
        if not pixi_toml.is_file():
            return {}
        try:
            with pixi_toml.open("rb") as handle:
                parsed = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        tasks = parsed.get("tasks", {})
        return tasks if isinstance(tasks, dict) else {}

    def run_task(self, task: str, timeout_seconds: int | None = None) -> CommandResult:
        command = self.build_task_command(task)
        timeout = timeout_seconds or self.settings.command_timeout_seconds
        try:
            completed = subprocess.run(
                command,
                cwd=self.settings.pybullet_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = self._normalize_timeout_stream(exc.output)
            stderr_parts = [f"Timed out after {timeout} seconds"]
            stderr = self._normalize_timeout_stream(exc.stderr)
            if stderr:
                stderr_parts.append(stderr)
            return CommandResult(
                command=tuple(command),
                returncode=124,
                stdout=stdout,
                stderr="\n".join(stderr_parts),
            )
        return CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    @staticmethod
    def _normalize_timeout_stream(stream: str | bytes | None) -> str:
        if stream is None:
            return ""
        if isinstance(stream, bytes):
            return stream.decode(errors="replace")
        return stream
