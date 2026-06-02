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
