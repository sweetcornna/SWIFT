from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_training_settings  # noqa: E402
from swift.experiments.multi_seed_checkpoint_evaluator import (  # noqa: E402
    MultiSeedCheckpointEvaluationConfig,
    run_multi_seed_checkpoint_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate multi-seed SWIFT checkpoints on holdout seeds.")
    parser.add_argument("--input", type=Path, required=True, help="Multi-seed training report JSON.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "training.yaml",
        help="Training settings YAML used for the holdout environment.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=[10000, 11000, 12000])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "multi_seed_checkpoint_holdout.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    settings = load_training_settings(args.config)
    config = MultiSeedCheckpointEvaluationConfig(
        input_report=args.input,
        output=args.output,
        holdout_seeds=tuple(args.holdout_seeds),
        episodes=args.episodes,
        environment=settings.environment,
    )
    if args.dry_run:
        print(f"SWIFT multi-seed checkpoint holdout config OK: {args.input}")
        return 0

    report = run_multi_seed_checkpoint_evaluation(config)
    checkpoint_count = sum(int(variant["holdout_checkpoint_count"]) for variant in report["variants"])
    print(f"SWIFT multi-seed checkpoint holdout written: {args.output}")
    print(
        f"variants={len(report['variants'])} "
        f"training_seeds={report['seed_count']} "
        f"checkpoints={checkpoint_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
