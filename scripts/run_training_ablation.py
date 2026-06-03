from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_training_settings  # noqa: E402
from swift.experiments.training_ablation_runner import (  # noqa: E402
    TrainingAblationConfig,
    run_training_ablation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SWIFT PPO MLP/HCA/HCA+APF CPU ablation tuning.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evidence-level", default="cpu_smoke_ablation")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "training" / "training_ablation.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    settings = load_training_settings(args.config)
    if args.dry_run:
        print(f"SWIFT training ablation config OK: {args.config}")
        return 0

    report = run_training_ablation(
        TrainingAblationConfig(
            settings=settings,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            output=args.output,
            evidence_level=args.evidence_level,
        )
    )
    print(f"SWIFT training ablation written: {args.output}")
    print(f"best_variant={report['best_variant']} variants={len(report['variants'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
