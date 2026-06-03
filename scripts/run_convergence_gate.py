from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.experiments.convergence_gate import (  # noqa: E402
    ConvergenceGateConfig,
    ConvergenceThresholds,
    run_convergence_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether a SWIFT training report may claim convergence.")
    parser.add_argument("--input", type=Path, required=True, help="Training or ablation report JSON to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "training" / "convergence_gate.json",
    )
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--max-collision-rate", type=float, default=0.0)
    parser.add_argument("--max-timeout-rate", type=float, default=0.05)
    parser.add_argument("--min-total-timesteps", type=int, default=4096)
    parser.add_argument("--min-episodes-completed", type=int, default=10)
    parser.add_argument("--required-evidence-level", default="long_training_convergence")
    args = parser.parse_args(argv)

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=args.input,
            output=args.output,
            thresholds=ConvergenceThresholds(
                min_success_rate=args.min_success_rate,
                max_collision_rate=args.max_collision_rate,
                max_timeout_rate=args.max_timeout_rate,
                min_total_timesteps=args.min_total_timesteps,
                min_episodes_completed=args.min_episodes_completed,
                required_evidence_level=args.required_evidence_level,
            ),
        )
    )
    print(f"SWIFT convergence gate written: {args.output}")
    print(
        f"convergence_claim={report['readiness']['convergence_claim']} "
        f"passed_gates={report['readiness']['passed_gates']}/{report['readiness']['required_gates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
