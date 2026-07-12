from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings
from swift.hover import ExternalHoverSubstrate, load_hover_training_config, run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SWIFT stabilized hover training on its external PyBullet substrate.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pybullet_hover.yaml")
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs/simulation.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/pybullet_hover/training")
    parser.add_argument("--run-name")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_hover_training_config(args.config)
    simulation = load_simulation_settings(args.simulation_config)
    substrate = ExternalHoverSubstrate(simulation.pybullet_root, simulation.pixi_executable)
    hashes = substrate.validate(require_runtime=not args.dry_run)
    if args.dry_run:
        print(f"SWIFT stabilized hover training config OK: profile={config.action_profile} substrate={substrate.root} source_files={len(hashes)}")
        return 0
    summary = run_training(substrate, config, args.output_root, run_name=args.run_name, timeout=args.timeout)
    print(Path(args.output_root).resolve() / summary["run_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
