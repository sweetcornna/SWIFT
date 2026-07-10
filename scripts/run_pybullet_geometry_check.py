from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings, load_training_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate randomized PyBullet final-phase geometry.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=ROOT / "configs" / "training_pybullet_randomized.yaml",
    )
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "pybullet_curriculum_geometry_validation.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args(argv)

    training_settings = load_training_settings(args.training_config)
    simulation_settings = load_simulation_settings(args.simulation_config)
    Config, run = _load_validator()
    config = Config(
        training_settings=training_settings,
        simulation_settings=simulation_settings,
        seed=args.seed,
        scenarios=args.scenarios,
        output=args.output,
    )
    if args.dry_run:
        print(
            "SWIFT randomized PyBullet geometry config OK: "
            f"seed={config.seed} scenarios={config.scenarios} output={config.output}"
        )
        return 0

    PyBulletRuntimeUnavailableError = _load_runtime_unavailable_error()
    try:
        report = run(config)
    except PyBulletRuntimeUnavailableError as exc:
        print(f"PyBullet runtime unavailable: {exc}")
        return 2
    print(f"SWIFT randomized PyBullet geometry written: {report['artifacts']['summary_json']}")
    print(
        f"geometry_valid={report['readiness']['geometry_valid']} "
        f"invalid_initial_clearance_count={report['metrics']['invalid_initial_clearance_count']} "
        f"first_step_collision_count={report['metrics']['first_step_collision_count']}"
    )
    if args.fail_on_invalid and not report["readiness"]["geometry_valid"]:
        return 1
    return 0


def _load_validator():
    from swift.experiments.pybullet_geometry_validator import (
        PyBulletGeometryValidationConfig,
        run_pybullet_geometry_validation,
    )

    return PyBulletGeometryValidationConfig, run_pybullet_geometry_validation


def _load_runtime_unavailable_error():
    from swift.sim import PyBulletRuntimeUnavailableError

    return PyBulletRuntimeUnavailableError


if __name__ == "__main__":
    raise SystemExit(main())
