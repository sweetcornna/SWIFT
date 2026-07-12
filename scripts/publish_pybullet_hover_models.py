from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.hover import publish_120k_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the curated three-seed 120k stabilized-hover model bundle.")
    parser.add_argument("--training-run", action="append", required=True, type=Path)
    parser.add_argument("--evaluation", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/robust-hover/120k")
    args = parser.parse_args(argv)
    if len(args.training_run) != 3 or len(args.evaluation) != 3:
        parser.error("exactly three --training-run and three --evaluation arguments are required")
    manifest = publish_120k_runs(args.training_run, args.evaluation, args.output)
    print(f"{args.output.resolve()} seeds={','.join(str(item['seed']) for item in manifest['seeds'])} files={len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
