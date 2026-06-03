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
    parser = argparse.ArgumentParser(description="Run a PPO+MLP training job on the PyBullet velocity substrate.")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs" / "training.yaml")
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--enable-obstacles", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional summary JSON path. Omit to use a run-id artifact path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    training_settings = load_training_settings(args.training_config)
    simulation_settings = load_simulation_settings(args.simulation_config)
    if args.dry_run:
        print(
            "SWIFT PyBullet PPO training config OK: "
            f"training={args.training_config} simulation={args.simulation_config}"
        )
        return 0

    PyBulletPPOTrainingRunConfig, run_pybullet_ppo_training = _load_runner()
    run_config = PyBulletPPOTrainingRunConfig(
        training_settings=training_settings,
        simulation_settings=simulation_settings,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        output=args.output,
        enable_pybullet_obstacles=args.enable_obstacles,
    )

    PyBulletRuntimeUnavailableError = _load_runtime_unavailable_error()
    try:
        result = run_pybullet_ppo_training(run_config)
    except PyBulletRuntimeUnavailableError as exc:
        print(f"PyBullet runtime unavailable: {exc}")
        return 2

    print(f"SWIFT PyBullet PPO training written: {result['artifacts']['summary_json']}")
    print(
        f"updates={result['training']['updates']} total_timesteps={result['training']['total_timesteps']} "
        f"runtime_contract={result['runtime']['runtime_contract']}"
    )
    return 0


def _load_runner():
    from swift.experiments.pybullet_training_runner import (
        PyBulletPPOTrainingRunConfig,
        run_pybullet_ppo_training,
    )

    return PyBulletPPOTrainingRunConfig, run_pybullet_ppo_training


def _load_runtime_unavailable_error():
    from swift.sim import PyBulletRuntimeUnavailableError

    return PyBulletRuntimeUnavailableError


if __name__ == "__main__":
    raise SystemExit(main())
