from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings
from swift.hover import ExternalHoverSubstrate, run_visualization


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only stabilized-hover visualizations from completed artifacts.")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs/simulation.yaml")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    simulation = load_simulation_settings(args.simulation_config)
    substrate = ExternalHoverSubstrate(simulation.pybullet_root, simulation.pixi_executable)
    substrate.validate(require_runtime=not args.dry_run)
    for source in args.input:
        if not source.resolve().is_dir():
            parser.error(f"input directory does not exist: {source}")
    if args.output.resolve().exists():
        parser.error(f"output must not already exist: {args.output}")
    if args.dry_run:
        print(f"SWIFT read-only hover visualization inputs OK: inputs={len(args.input)} output={args.output.resolve()}")
        return 0
    print(run_visualization(substrate, args.input, args.output, html=args.html, timeout=args.timeout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
