from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swift.config import load_simulation_settings  # noqa: E402
from swift.core import DroneAction  # noqa: E402
from swift.sim import PyBulletRuntimeUnavailableError, PyBulletVelocityRuntimeEnv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an optional headless PyBullet runtime adapter smoke.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--runtime", choices=("auto", "direct", "pixi"), default="auto")
    args = parser.parse_args(argv)

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    settings = load_simulation_settings(args.config)
    if args.runtime == "pixi":
        return _run_pixi_runtime_smoke(settings, args.steps)

    try:
        env = PyBulletVelocityRuntimeEnv(settings)
    except PyBulletRuntimeUnavailableError as exc:
        if args.runtime == "auto":
            print(f"PyBullet direct runtime unavailable: {exc}")
            return _run_pixi_runtime_smoke(settings, args.steps)
        print(f"PyBullet runtime unavailable: {exc}")
        return 2

    try:
        observation, _ = env.reset(seed=0)
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(args.steps):
            observation, reward, terminated, truncated, _ = env.step(
                DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
            )
    finally:
        env.close()

    print("SWIFT PyBullet runtime smoke: OK")
    print(f"observation_dim={len(observation)} reward={reward} terminated={terminated} truncated={truncated}")
    return 0


def _run_pixi_runtime_smoke(settings, steps: int) -> int:
    if not settings.pixi_executable.is_file():
        print(f"PyBullet runtime unavailable: Pixi executable not found: {settings.pixi_executable}")
        return 2
    code = _pixi_smoke_code(settings.pybullet_root, steps)
    completed = subprocess.run(
        [str(settings.pixi_executable), "run", "python", "-c", code],
        cwd=settings.pybullet_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode != 0:
        return completed.returncode
    print("SWIFT PyBullet runtime smoke: OK")
    return 0


def _pixi_smoke_code(pybullet_root: Path, steps: int) -> str:
    return f"""
import sys
from pathlib import Path
import numpy as np

root = Path(r"{pybullet_root}")
sys.path.insert(0, str(root / "external" / "gym-pybullet-drones"))

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

env = VelocityAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    gui=False,
    record=False,
    obstacles=False,
    user_debug_gui=False,
)
try:
    obs, _ = env.reset(seed=0)
    reward = 0.0
    terminated = False
    truncated = False
    action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    for _ in range({int(steps)}):
        obs, reward, terminated, truncated, _ = env.step(action)
    row = obs[0]
    swift_obs = tuple(row[0:3]) + tuple(row[10:13]) + (row[9],) + (0.0,) * 8
    print(f"observation_dim={{len(swift_obs)}} raw_observation_dim={{int(obs.shape[-1])}} reward={{float(reward)}} terminated={{bool(terminated)}} truncated={{bool(truncated)}}")
finally:
    env.close()
"""


if __name__ == "__main__":
    raise SystemExit(main())
