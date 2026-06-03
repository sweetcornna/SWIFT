from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.experiments.stage1_report import write_stage1_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a strict SWIFT Stage 1 training and tuning report.")
    parser.add_argument(
        "--ppo-summary",
        type=Path,
        default=ROOT / "outputs" / "training" / "ppo_smoke.json",
        help="PPO training summary JSON path.",
    )
    parser.add_argument(
        "--tuning-summary",
        type=Path,
        default=ROOT / "outputs" / "tuning" / "stage1_grid.json",
        help="Stage 1 tuning summary JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "reports" / "stage1_training_tuning_report.json",
        help="Report output JSON path.",
    )
    args = parser.parse_args(argv)

    report = write_stage1_report(args.ppo_summary, args.tuning_summary, args.output)
    print(f"SWIFT Stage 1 report written: {args.output}")
    print(
        "readiness="
        f"{report['readiness']['collision_to_success']} "
        f"training_updates={report['training']['updates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
