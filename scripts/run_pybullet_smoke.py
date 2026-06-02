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

    print("SWIFT pybullet substrate: OK")
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
