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
from swift.core import DroneAction, ObstacleState  # noqa: E402
from swift.envs import PyBulletVelocityTrainingEnv, SimpleAvoidanceSettings  # noqa: E402
from swift.sim import PyBulletRuntimeUnavailableError, PyBulletVelocityRuntimeEnv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an optional headless PyBullet runtime adapter smoke.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "simulation.yaml")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--runtime", choices=("auto", "direct", "pixi"), default="auto")
    parser.add_argument("--enable-obstacles", action="store_true")
    parser.add_argument("--training-env", action="store_true", help="Smoke the SWIFT PPO-compatible training wrapper.")
    args = parser.parse_args(argv)

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    settings = load_simulation_settings(args.config)
    if args.runtime == "pixi":
        return _run_pixi_runtime_smoke(settings, args.steps, args.enable_obstacles, training_env=args.training_env)

    if args.training_env:
        status = _run_direct_training_env_smoke(settings, args.steps, args.enable_obstacles)
        if status == 2 and args.runtime == "auto":
            return _run_pixi_runtime_smoke(settings, args.steps, args.enable_obstacles, training_env=True)
        return status

    try:
        env = PyBulletVelocityRuntimeEnv(settings, enable_obstacles=args.enable_obstacles)
    except PyBulletRuntimeUnavailableError as exc:
        if args.runtime == "auto":
            print(f"PyBullet direct runtime unavailable: {exc}")
            return _run_pixi_runtime_smoke(settings, args.steps, args.enable_obstacles, training_env=args.training_env)
        print(f"PyBullet runtime unavailable: {exc}")
        return 2

    try:
        observation, info = env.reset(seed=0)
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(
                DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
            )
    finally:
        env.close()

    print("SWIFT PyBullet runtime smoke: OK")
    print(
        f"observation_dim={len(observation)} reward={reward} terminated={terminated} "
        f"truncated={truncated} obstacles_enabled={bool(args.enable_obstacles)} "
        f"collided={bool(info.get('collided', False))} contact_count={int(info.get('contact_count', 0))}"
    )
    return 0


def _run_direct_training_env_smoke(settings, steps: int, enable_obstacles: bool = False) -> int:
    env_settings = _training_env_settings(enable_obstacles)
    try:
        env = PyBulletVelocityTrainingEnv(
            simulation_settings=settings,
            settings=env_settings,
            enable_pybullet_obstacles=enable_obstacles,
        )
    except PyBulletRuntimeUnavailableError as exc:
        print(f"PyBullet runtime unavailable: {exc}")
        return 2

    try:
        observation, info = env.reset(seed=0)
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(steps):
            observation, reward, terminated, truncated, info = env.step(
                DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
            )
    finally:
        env.close()

    print("SWIFT PyBullet training env smoke: OK")
    print(_training_env_summary(observation, reward, terminated, truncated, info, enable_obstacles, env_settings))
    return 0


def _run_pixi_runtime_smoke(settings, steps: int, enable_obstacles: bool = False, *, training_env: bool = False) -> int:
    if not settings.pixi_executable.is_file():
        print(f"PyBullet runtime unavailable: Pixi executable not found: {settings.pixi_executable}")
        return 2
    code = (
        _pixi_training_env_smoke_code(ROOT, settings.pybullet_root, steps, enable_obstacles=enable_obstacles)
        if training_env
        else _pixi_smoke_code(settings.pybullet_root, steps, enable_obstacles=enable_obstacles)
    )
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
    print("SWIFT PyBullet training env smoke: OK" if training_env else "SWIFT PyBullet runtime smoke: OK")
    return 0


def _training_env_settings(enable_obstacles: bool) -> SimpleAvoidanceSettings:
    obstacles = (
        (ObstacleState(position=(2.0, 0.0, 1.0), radius=0.3),)
        if enable_obstacles
        else ()
    )
    return SimpleAvoidanceSettings(
        goal=(4.0, 0.0, 1.0),
        obstacles=obstacles,
        max_steps=8,
        max_speed=1.0,
        max_climb_rate=0.4,
        safety_margin=0.1,
    )


def _training_env_summary(
    observation,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    enable_obstacles: bool,
    settings: SimpleAvoidanceSettings,
) -> str:
    return (
        f"observation_dim={len(observation)} reward={float(reward)} terminated={bool(terminated)} "
        f"truncated={bool(truncated)} runtime_contract={info.get('runtime_contract', '')} "
        f"minimum_safety_distance={float(info.get('minimum_safety_distance', 0.0))} "
        f"nearest_obstacle_radius={float(observation[13])} obstacles_enabled={bool(enable_obstacles)} "
        f"swift_obstacles={len(settings.obstacles)}"
    )


def _pixi_smoke_code(pybullet_root: Path, steps: int, enable_obstacles: bool = False) -> str:
    obstacles_literal = "True" if enable_obstacles else "False"
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
    obstacles={obstacles_literal},
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
    print(f"observation_dim={{len(swift_obs)}} raw_observation_dim={{int(obs.shape[-1])}} reward={{float(reward)}} terminated={{bool(terminated)}} truncated={{bool(truncated)}} obstacles_enabled={obstacles_literal}")
finally:
    env.close()
"""


def _pixi_training_env_smoke_code(
    swift_root: Path,
    pybullet_root: Path,
    steps: int,
    enable_obstacles: bool = False,
) -> str:
    obstacles_literal = "True" if enable_obstacles else "False"
    obstacle_tuple = (
        "(ObstacleState(position=(2.0, 0.0, 1.0), radius=0.3),)"
        if enable_obstacles
        else "()"
    )
    swift_obstacle_count = 1 if enable_obstacles else 0
    return f"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

swift_root = Path(r"{swift_root}")
pybullet_root = Path(r"{pybullet_root}")
sys.path.insert(0, str(swift_root / "src"))
sys.path.insert(0, str(pybullet_root / "external" / "gym-pybullet-drones"))

config_module = types.ModuleType("swift.config")
class SimulationSettings:
    pass
config_module.SimulationSettings = SimulationSettings
sys.modules.setdefault("swift.config", config_module)

from swift.core import DroneAction, ObstacleState
from swift.envs.pybullet_velocity import PyBulletVelocityTrainingEnv
from swift.envs.simple_avoidance import SimpleAvoidanceSettings

settings = SimpleNamespace(pybullet_root=pybullet_root)
env_settings = SimpleAvoidanceSettings(
    goal=(4.0, 0.0, 1.0),
    obstacles={obstacle_tuple},
    max_steps=8,
    max_speed=1.0,
    max_climb_rate=0.4,
    safety_margin=0.1,
)
env = PyBulletVelocityTrainingEnv(
    simulation_settings=settings,
    settings=env_settings,
    enable_pybullet_obstacles={obstacles_literal},
    obstacle_randomization=SimpleNamespace(enabled=False),
    reward_settings=SimpleNamespace(
        arrival_reward=100.0,
        approach_scale=1.0,
        collision_penalty=100.0,
        timeout_penalty=0.0,
        episode_time_penalty=1.0,
        heading_smoothness_penalty=0.05,
    ),
    curriculum=SimpleNamespace(enabled=False),
)
try:
    observation, info = env.reset(seed=0)
    reward = 0.0
    terminated = False
    truncated = False
    for _ in range({int(steps)}):
        observation, reward, terminated, truncated, info = env.step(
            DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
        )
    expected_contract = "pybullet_velocity_training_compatibility"
    if info.get("runtime_contract", "") != expected_contract:
        raise RuntimeError(f"unexpected runtime_contract={{info.get('runtime_contract', '')}}")
    print(f"observation_dim={{len(observation)}} reward={{float(reward)}} terminated={{bool(terminated)}} truncated={{bool(truncated)}} runtime_contract=pybullet_velocity_training_compatibility minimum_safety_distance={{float(info.get('minimum_safety_distance', 0.0))}} nearest_obstacle_radius={{float(observation[13])}} obstacles_enabled={obstacles_literal} swift_obstacles={swift_obstacle_count}")
finally:
    env.close()
"""


if __name__ == "__main__":
    raise SystemExit(main())
