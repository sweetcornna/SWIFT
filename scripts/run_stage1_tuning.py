from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.experiments.tuning_runner import run_stage1_policy_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic SWIFT Stage 1 policy tuning.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "stage1" / "stage1_tuning.json",
        help="Path for the JSON tuning output.",
    )
    args = parser.parse_args(argv)

    result = run_stage1_policy_search()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"SWIFT stage1 tuning written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
