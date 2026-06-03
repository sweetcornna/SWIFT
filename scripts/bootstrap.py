from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap SWIFT on a new machine.")
    parser.add_argument("--mode", choices=("quick", "train", "pybullet"), default="quick")
    parser.add_argument("--venv-path", type=Path, default=ROOT / ".venv")
    parser.add_argument("--pybullet-root", type=Path, default=None)
    parser.add_argument("--pybullet-venv", type=Path, default=ROOT / ".venv-pybullet")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for repo-native venvs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--no-verify", action="store_true", help="Install only; skip verification commands.")
    args = parser.parse_args(argv)

    commands = _plan(args)
    print(f"SWIFT bootstrap mode={args.mode} dry_run={bool(args.dry_run)}")
    for command in commands:
        print(_format_command(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    print("SWIFT bootstrap complete")
    return 0


def _plan(args: argparse.Namespace) -> list[list[str]]:
    if args.mode == "pybullet":
        return _pybullet_plan(args)
    return _repo_native_plan(args)


def _repo_native_plan(args: argparse.Namespace) -> list[list[str]]:
    venv_python = _venv_python(args.venv_path)
    extra = ".[dev,train]" if args.mode == "train" else ".[dev]"
    commands = [
        [args.python, "-m", "venv", str(args.venv_path)],
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(venv_python), "-m", "pip", "install", "-e", extra],
    ]
    if args.no_verify:
        return commands

    commands.extend(
        [
            [str(venv_python), "-m", "pytest"],
            [str(venv_python), str(ROOT / "scripts" / "swift_healthcheck.py")],
            [
                str(venv_python),
                str(ROOT / "scripts" / "run_baseline_demo.py"),
                "--episodes",
                "3",
                "--output",
                str(ROOT / "outputs" / "baseline" / "baseline_metrics.json"),
            ],
        ]
    )
    if args.mode == "train":
        commands.append(
            [
                str(venv_python),
                str(ROOT / "scripts" / "run_ppo_mlp_smoke.py"),
                "--total-timesteps",
                "128",
                "--output",
                str(ROOT / "outputs" / "training" / "ppo_smoke.json"),
            ]
        )
    return commands


def _pybullet_plan(args: argparse.Namespace) -> list[list[str]]:
    pybullet_root = args.pybullet_root or _default_pybullet_root()
    pixi_python = pybullet_root / ".pixi" / "envs" / "default" / _python_executable_name()
    venv_python = _venv_python(args.pybullet_venv)
    commands = [
        [str(pixi_python), "-m", "venv", "--system-site-packages", str(args.pybullet_venv)],
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(venv_python), "-m", "pip", "install", "-e", str(ROOT), "torch", "PyYAML"],
    ]
    if args.no_verify:
        return commands

    commands.extend(
        [
            [str(venv_python), str(ROOT / "scripts" / "run_pybullet_smoke.py"), "--check-only"],
            [
                str(venv_python),
                str(ROOT / "scripts" / "run_pybullet_runtime_smoke.py"),
                "--runtime",
                "pixi",
                "--training-env",
                "--steps",
                "1",
            ],
        ]
    )
    return commands


def _default_pybullet_root() -> Path:
    config = ROOT / "configs" / "simulation.yaml"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("pybullet_root:"):
                return Path(line.split(":", 1)[1].strip())
    return Path(r"D:\project\pybullet") if os.name == "nt" else ROOT.parent / "pybullet"


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _python_executable_name() -> str:
    return "python.exe" if os.name == "nt" else "python"


def _format_command(command: list[str]) -> str:
    return " ".join(_quote(part) for part in command)


def _quote(value: str) -> str:
    if not value:
        return '""'
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
