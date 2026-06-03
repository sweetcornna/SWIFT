from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_training_settings  # noqa: E402
from swift.experiments.multi_seed_training_runner import (  # noqa: E402
    MultiSeedTrainingConfig,
    run_multi_seed_training,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SWIFT multi-seed PPO MLP/HCA/HCA+APF training ablation.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evidence-level", default="cpu_smoke_ablation")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "training" / "multi_seed_training.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    settings = load_training_settings(args.config)
    if args.dry_run:
        MultiSeedTrainingConfig(
            settings=settings,
            seeds=tuple(args.seeds),
            total_timesteps=args.total_timesteps,
            output=args.output,
            evidence_level=args.evidence_level,
        )
        print(f"SWIFT multi-seed training config OK: {args.config}")
        return 0

    report = run_multi_seed_training(
        MultiSeedTrainingConfig(
            settings=settings,
            seeds=tuple(args.seeds),
            total_timesteps=args.total_timesteps,
            output=args.output,
            evidence_level=args.evidence_level,
        )
    )
    print(f"SWIFT multi-seed training written: {args.output}")
    print(f"seed_count={report['seed_count']} variants={len(report['variants'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
