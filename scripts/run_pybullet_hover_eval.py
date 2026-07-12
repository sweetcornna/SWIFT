from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings
from swift.hover import ExternalHoverSubstrate, load_bound_training_summary, run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only SWIFT evaluation of a bound stabilized-hover checkpoint.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs/simulation.yaml")
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    simulation = load_simulation_settings(args.simulation_config)
    substrate = ExternalHoverSubstrate(simulation.pybullet_root, simulation.pixi_executable)
    substrate.validate(require_runtime=not args.dry_run)
    load_bound_training_summary(args.model.resolve().parent, required_models=(args.model.name,))
    if args.dry_run:
        print(f"SWIFT stabilized hover evaluation inputs OK: model={args.model.resolve()}")
        return 0
    report = run_evaluation(substrate, args.model, args.output, evaluation_seed=args.evaluation_seed, cases=args.cases, device=args.device, timeout=args.timeout)
    print(f"{args.output.resolve()} verdict={report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
