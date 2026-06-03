from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.experiments.ppo_checkpoint_evaluator import (  # noqa: E402
    PPOCheckpointEvaluationConfig,
    run_ppo_checkpoint_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a SWIFT PPO checkpoint deterministically.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PPO checkpoint path.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of deterministic episodes to run.")
    parser.add_argument("--seed", type=int, default=0, help="Base evaluation seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "ppo_checkpoint_eval.json",
        help="Evaluation summary JSON path.",
    )
    args = parser.parse_args(argv)

    summary = run_ppo_checkpoint_evaluation(
        PPOCheckpointEvaluationConfig(
            checkpoint_path=args.checkpoint,
            output=args.output,
            episodes=args.episodes,
            seed=args.seed,
        )
    )
    print(f"SWIFT PPO checkpoint evaluation written: {args.output}")
    print(
        f"episodes={summary['episodes_requested']} "
        f"success_rate={summary['metrics']['success_rate']} "
        f"collision_rate={summary['metrics']['collision_rate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
