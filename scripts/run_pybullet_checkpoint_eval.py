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
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint on the PyBullet velocity substrate.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PPO checkpoint path.")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs" / "training.yaml")
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-obstacles", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    training_settings = load_training_settings(args.training_config)
    simulation_settings = load_simulation_settings(args.simulation_config)
    if args.dry_run:
        print(
            "SWIFT PyBullet checkpoint evaluation config OK: "
            f"checkpoint={args.checkpoint} training={args.training_config} simulation={args.simulation_config}"
        )
        return 0

    PyBulletCheckpointEvaluationConfig, run_pybullet_checkpoint_evaluation = _load_evaluator()
    config = PyBulletCheckpointEvaluationConfig(
        checkpoint_path=args.checkpoint,
        training_settings=training_settings,
        simulation_settings=simulation_settings,
        training_config_path=args.training_config,
        simulation_config_path=args.simulation_config,
        output=args.output,
        episodes=args.episodes,
        seed=args.seed,
        enable_pybullet_obstacles=args.enable_obstacles,
    )
    PyBulletRuntimeUnavailableError = _load_runtime_unavailable_error()
    try:
        summary = run_pybullet_checkpoint_evaluation(config)
    except PyBulletRuntimeUnavailableError as exc:
        print(f"PyBullet runtime unavailable: {exc}")
        return 2

    print(f"SWIFT PyBullet checkpoint evaluation written: {summary['artifacts']['summary_json']}")
    print(
        f"episodes={summary['episodes_requested']} "
        f"success_rate={summary['metrics']['success_rate']} "
        f"collision_rate={summary['metrics']['collision_rate']} "
        f"runtime_contract={summary['runtime']['runtime_contract']}"
    )
    return 0


def _load_evaluator():
    from swift.experiments.pybullet_checkpoint_evaluator import (
        PyBulletCheckpointEvaluationConfig,
        run_pybullet_checkpoint_evaluation,
    )

    return PyBulletCheckpointEvaluationConfig, run_pybullet_checkpoint_evaluation


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        parser.error("checkpoint must exist")
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if args.seed < 0:
        parser.error("seed must be non-negative")


def _load_runtime_unavailable_error():
    from swift.sim import PyBulletRuntimeUnavailableError

    return PyBulletRuntimeUnavailableError


if __name__ == "__main__":
    raise SystemExit(main())
