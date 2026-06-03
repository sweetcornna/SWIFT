from pathlib import Path
from types import SimpleNamespace

import pytest

from swift.config import SimulationSettings
from swift.core import DroneAction
from swift.sim.pybullet_runtime import (
    PyBulletRuntimeUnavailableError,
    PyBulletVelocityRuntimeEnv,
    build_pybullet_drones_path,
    drone_action_to_velocity_command,
)


def make_settings(root: Path) -> SimulationSettings:
    return SimulationSettings(
        pybullet_root=root,
        pixi_executable=root / ".tools" / "pixi" / "pixi.exe",
        required_tasks=("test", "drone-demo"),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )


def test_build_pybullet_drones_path_uses_external_substrate_root(tmp_path: Path):
    expected = tmp_path / "external" / "gym-pybullet-drones"

    assert build_pybullet_drones_path(make_settings(tmp_path)) == expected


def test_runtime_env_rejects_missing_vendored_path(tmp_path: Path):
    with pytest.raises(PyBulletRuntimeUnavailableError, match="gym-pybullet-drones"):
        PyBulletVelocityRuntimeEnv(make_settings(tmp_path))


def test_drone_action_to_velocity_command_maps_heading_and_climb_rate():
    command = drone_action_to_velocity_command(
        DroneAction(speed=2.0, heading_delta=0.0, climb_rate=0.5),
        max_speed=4.0,
    )

    assert command[0] == pytest.approx([1.0, 0.0, 0.25, 0.5])


def test_runtime_env_wraps_velocity_aviary_contract(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_env = FakeVelocityAviary()

    runtime = PyBulletVelocityRuntimeEnv(
        make_settings(tmp_path),
        velocity_aviary_cls=lambda **_: fake_env,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    observation, info = runtime.reset(seed=7)
    next_observation, reward, terminated, truncated, step_info = runtime.step(
        DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
    )
    runtime.close()

    assert len(observation) == 15
    assert len(next_observation) == 15
    assert reward == -1.0
    assert terminated is False
    assert truncated is False
    assert info["backend"] == "pybullet_velocity_aviary"
    assert step_info["backend"] == "pybullet_velocity_aviary"
    assert fake_env.reset_seed == 7
    assert fake_env.last_action[0] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert fake_env.closed is True


class FakeVelocityAviary:
    def __init__(self) -> None:
        self.reset_seed = None
        self.last_action = None
        self.closed = False

    def reset(self, seed=None, options=None):
        self.reset_seed = seed
        return [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 10, 11, 12, 13]], {
            "source": "fake"
        }

    def step(self, action):
        self.last_action = action
        return (
            [[1.1, 2.2, 3.3, 0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.0, 0.0, 0.0, 14, 15, 16, 17]],
            -1,
            False,
            False,
            {"source": "fake-step"},
        )

    def close(self):
        self.closed = True
