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
from swift.config import ConvergenceGateProfile, load_evaluation_settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether a SWIFT training report may claim convergence.")
    parser.add_argument("--input", type=Path, required=True, help="Training or ablation report JSON to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "training" / "convergence_gate.json",
    )
    parser.add_argument("--config", type=Path, help="Evaluation settings YAML with convergence gate profiles.")
    parser.add_argument("--profile", help="Convergence gate profile name from the evaluation settings YAML.")
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--max-collision-rate", type=float)
    parser.add_argument("--max-timeout-rate", type=float)
    parser.add_argument("--min-total-timesteps", type=int)
    parser.add_argument("--min-episodes-completed", type=int)
    parser.add_argument("--required-evidence-level")
    parser.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="Exit with status 1 when the convergence gate rejects the input report.",
    )
    args = parser.parse_args(argv)

    report = run_convergence_gate(
        ConvergenceGateConfig(
            input_report=args.input,
            output=args.output,
            thresholds=_thresholds_from_args(args),
        )
    )
    print(f"SWIFT convergence gate written: {args.output}")
    print(
        f"convergence_claim={report['readiness']['convergence_claim']} "
        f"passed_gates={report['readiness']['passed_gates']}/{report['readiness']['required_gates']}"
    )
    if args.fail_on_reject and not report["readiness"]["convergence_claim"]:
        return 1
    return 0


def _thresholds_from_args(args: argparse.Namespace) -> ConvergenceThresholds:
    profile = _profile_from_args(args)
    return ConvergenceThresholds(
        convergence_claim_allowed=profile.convergence_claim_allowed,
        min_success_rate=args.min_success_rate
        if args.min_success_rate is not None
        else profile.min_success_rate,
        max_collision_rate=args.max_collision_rate
        if args.max_collision_rate is not None
        else profile.max_collision_rate,
        max_timeout_rate=args.max_timeout_rate if args.max_timeout_rate is not None else profile.max_timeout_rate,
        min_total_timesteps=args.min_total_timesteps
        if args.min_total_timesteps is not None
        else profile.min_total_timesteps,
        min_episodes_completed=args.min_episodes_completed
        if args.min_episodes_completed is not None
        else profile.min_episodes_completed,
        required_evidence_level=args.required_evidence_level
        if args.required_evidence_level is not None
        else profile.required_evidence_level,
        required_variants=profile.required_variants,
    )


def _profile_from_args(args: argparse.Namespace) -> ConvergenceGateProfile:
    if args.profile or args.config:
        settings = load_evaluation_settings(args.config or ROOT / "configs" / "evaluation.yaml")
        return settings.convergence_profile(args.profile)
    return ConvergenceGateProfile()


if __name__ == "__main__":
    raise SystemExit(main())
