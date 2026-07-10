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
    parser = argparse.ArgumentParser(description="Run randomized PyBullet PPO training across multiple seeds.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=ROOT / "configs" / "training_pybullet_randomized.yaml",
    )
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10])
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "training" / "pybullet_randomized_3x150k.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    training_settings = load_training_settings(args.training_config)
    simulation_settings = load_simulation_settings(args.simulation_config)
    Config, run = _load_runner()
    config = Config(
        training_settings=training_settings,
        simulation_settings=simulation_settings,
        seeds=tuple(args.seeds),
        total_timesteps=args.total_timesteps,
        output=args.output,
    )
    if args.dry_run:
        total_timesteps = config.total_timesteps or training_settings.run.total_timesteps
        print(
            "SWIFT randomized PyBullet multi-seed training config OK: "
            f"seeds={','.join(str(seed) for seed in config.seeds)} "
            f"total_timesteps={total_timesteps} output={config.output}"
        )
        return 0

    PyBulletRuntimeUnavailableError = _load_runtime_unavailable_error()
    try:
        report = run(config)
    except PyBulletRuntimeUnavailableError as exc:
        print(f"PyBullet runtime unavailable: {exc}")
        return 2
    print(f"SWIFT randomized PyBullet multi-seed training written: {report['artifacts']['summary_json']}")
    print(
        f"seed_count={report['seed_count']} total_timesteps={report['total_timesteps']} "
        f"all_seeds_completed={report['readiness']['all_seeds_completed']}"
    )
    return 0


def _load_runner():
    from swift.experiments.pybullet_multi_seed_training_runner import (
        PyBulletMultiSeedTrainingConfig,
        run_pybullet_multi_seed_training,
    )

    return PyBulletMultiSeedTrainingConfig, run_pybullet_multi_seed_training


def _load_runtime_unavailable_error():
    from swift.sim import PyBulletRuntimeUnavailableError

    return PyBulletRuntimeUnavailableError


if __name__ == "__main__":
    raise SystemExit(main())
