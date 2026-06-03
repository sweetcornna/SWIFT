from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.experiments.multi_scenario_evaluator import (  # noqa: E402
    MultiScenarioEvaluationConfig,
    run_multi_scenario_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SWIFT Stage 4 deterministic multi-scenario evaluation.")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Limit displayed candidates per scenario; aggregation still uses the full search grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "stage4" / "multi_scenario_eval.json",
    )
    args = parser.parse_args(argv)

    report = run_multi_scenario_evaluation(
        MultiScenarioEvaluationConfig(
            candidate_limit=args.candidate_limit,
            output=args.output,
        )
    )
    print(f"SWIFT multi-scenario evaluation written: {args.output}")
    print(
        "scenario_count="
        f"{report['summary']['scenario_count']} "
        f"global_candidate_accepted={report['readiness']['global_candidate_accepted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
