from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply strict worst-case gates to randomized PyBullet holdout evidence.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "pybullet_randomized_3x100k_gate.json",
    )
    parser.add_argument("--min-training-seed-count", type=int, default=3)
    parser.add_argument("--min-total-timesteps", type=int, default=100000)
    parser.add_argument("--min-holdout-episodes", type=int, default=100)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--max-collision-rate", type=float, default=0.0)
    parser.add_argument("--max-timeout-rate", type=float, default=0.05)
    parser.add_argument("--min-average-safety-distance", type=float, default=0.10)
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args(argv)

    Config, Thresholds, run = _load_gate()
    report = run(
        Config(
            input_report=args.input,
            output=args.output,
            thresholds=Thresholds(
                min_training_seed_count=args.min_training_seed_count,
                min_total_timesteps=args.min_total_timesteps,
                min_holdout_episodes_per_checkpoint=args.min_holdout_episodes,
                min_success_rate=args.min_success_rate,
                max_collision_rate=args.max_collision_rate,
                max_timeout_rate=args.max_timeout_rate,
                min_average_minimum_safety_distance=args.min_average_safety_distance,
            ),
        )
    )
    robustness_claim = bool(report["readiness"]["robustness_claim"])
    print(f"SWIFT PyBullet robustness gate written: {report['artifacts']['summary_json']}")
    print(
        f"robustness_claim={robustness_claim} "
        f"passed_gates={report['readiness']['passed_gates']}/{report['readiness']['required_gates']}"
    )
    return 1 if args.fail_on_reject and not robustness_claim else 0


def _load_gate():
    from swift.experiments.pybullet_robustness_gate import (
        PyBulletRobustnessGateConfig,
        PyBulletRobustnessThresholds,
        run_pybullet_robustness_gate,
    )

    return PyBulletRobustnessGateConfig, PyBulletRobustnessThresholds, run_pybullet_robustness_gate


if __name__ == "__main__":
    raise SystemExit(main())
