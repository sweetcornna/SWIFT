from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_training_settings  # noqa: E402
from swift.experiments.hca_training_runner import HCATrainingRunConfig, run_hca_training_smoke  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a CPU PPO+HCA Stage 2 training smoke.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "training" / "ppo_hca_smoke.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    settings = load_training_settings(args.config)
    if args.dry_run:
        print(f"SWIFT PPO HCA smoke config OK: {args.config}")
        return 0

    result = run_hca_training_smoke(
        HCATrainingRunConfig(
            settings=settings,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            output=args.output,
        )
    )
    print(f"SWIFT PPO HCA smoke written: {args.output}")
    print(f"updates={result['training']['updates']} total_timesteps={result['training']['total_timesteps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
