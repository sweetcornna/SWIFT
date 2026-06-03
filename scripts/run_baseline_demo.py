from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.experiments import BaselineRunConfig, run_baseline_episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SWIFT Stage 1 baseline smoke demo.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of baseline episodes to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "baseline" / "baseline_metrics.json",
        help="Path for the JSON metrics output.",
    )
    args = parser.parse_args(argv)

    metrics = run_baseline_episodes(BaselineRunConfig(episodes=args.episodes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"SWIFT baseline metrics written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
