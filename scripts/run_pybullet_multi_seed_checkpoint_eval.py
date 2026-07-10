from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings, load_training_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate randomized PyBullet checkpoints on shared holdout layouts.")
    parser.add_argument("--input", type=Path, required=True, help="Multi-seed PyBullet training report.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=ROOT / "configs" / "training_pybullet_randomized.yaml",
    )
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--holdout-seed", type=int, default=1000000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "pybullet_randomized_3x100k_holdout.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _validate_input(parser, args.input)

    training_settings = load_training_settings(args.training_config)
    simulation_settings = load_simulation_settings(args.simulation_config)
    Config, run = _load_evaluator()
    config = Config(
        training_report_path=args.input,
        training_settings=training_settings,
        simulation_settings=simulation_settings,
        training_config_path=args.training_config,
        simulation_config_path=args.simulation_config,
        episodes_per_checkpoint=args.episodes,
        holdout_seed=args.holdout_seed,
        output=args.output,
    )
    if args.dry_run:
        print(
            "SWIFT randomized PyBullet holdout config OK: "
            f"episodes={config.episodes_per_checkpoint} holdout_seed={config.holdout_seed} "
            f"input={config.training_report_path}"
        )
        return 0

    PyBulletRuntimeUnavailableError = _load_runtime_unavailable_error()
    try:
        report = run(config)
    except PyBulletRuntimeUnavailableError as exc:
        print(f"PyBullet runtime unavailable: {exc}")
        return 2
    print(f"SWIFT randomized PyBullet holdout written: {report['artifacts']['summary_json']}")
    print(
        f"checkpoint_count={len(report['checkpoint_evaluations'])} "
        f"episodes_per_checkpoint={report['holdout']['episodes_per_checkpoint']} "
        f"worst_success_rate={report['metrics']['worst_success_rate']}"
    )
    return 0


def _validate_input(parser: argparse.ArgumentParser, path: Path) -> None:
    if not path.is_file():
        parser.error("input report must exist")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"input report must be valid JSON: {exc}")
    if report.get("record_type") != "pybullet_multi_seed_training_report":
        parser.error("input must be a pybullet_multi_seed_training_report")


def _load_evaluator():
    from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
        PyBulletMultiSeedCheckpointEvaluationConfig,
        run_pybullet_multi_seed_checkpoint_evaluation,
    )

    return PyBulletMultiSeedCheckpointEvaluationConfig, run_pybullet_multi_seed_checkpoint_evaluation


def _load_runtime_unavailable_error():
    from swift.sim import PyBulletRuntimeUnavailableError

    return PyBulletRuntimeUnavailableError


if __name__ == "__main__":
    raise SystemExit(main())
