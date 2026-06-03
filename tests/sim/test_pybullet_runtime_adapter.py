from pathlib import Path
from types import SimpleNamespace

import pytest

from swift.config import SimulationSettings
from swift.core import DroneAction
from swift.envs import PyBulletVelocityTrainingEnv, SimpleAvoidanceSettings
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


def test_pybullet_training_env_exposes_ppo_contract_with_fake_aviary(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_env = FakeVelocityAviary()
    settings = SimpleAvoidanceSettings(max_steps=2, max_speed=1.5, max_climb_rate=0.25)
    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        velocity_aviary_cls=lambda **_: fake_env,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    observation, info = training_env.reset(seed=11)
    step_one = training_env.step(DroneAction(speed=0.5, heading_delta=0.0, climb_rate=0.0))
    step_two = training_env.step(DroneAction(speed=0.5, heading_delta=0.0, climb_rate=0.0))
    training_env.close()

    assert training_env.settings == settings
    assert training_env.observation_shape == (15,)
    assert training_env.action_shape == (3,)
    assert len(observation) == 15
    assert len(step_one[0]) == 15
    assert len(step_two[0]) == 15
    assert info["backend"] == "pybullet_velocity_aviary"
    assert step_one[3] is False
    assert step_one[4]["timed_out"] is False
    assert step_two[2] is False
    assert step_two[3] is True
    assert step_two[4]["timed_out"] is True
    assert fake_env.closed is True


def test_pybullet_training_env_adds_goal_tail_reward_and_episode_metrics(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_env = FakeGoalVelocityAviary()
    settings = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.0),
        goal=(1.0, 0.0, 0.0),
        goal_radius=0.2,
        max_steps=4,
        max_speed=1.0,
    )
    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        velocity_aviary_cls=lambda **_: fake_env,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    observation, info = training_env.reset(seed=13)
    next_observation, reward, terminated, truncated, step_info = training_env.step(
        DroneAction(speed=1.0, heading_delta=0.0, climb_rate=0.0)
    )

    assert observation[7:10] == pytest.approx((1.0, 0.0, 0.0))
    assert observation[14] == pytest.approx(1.0)
    assert next_observation[7:10] == pytest.approx((0.05, 0.0, 0.0))
    assert next_observation[14] == pytest.approx(0.05)
    assert reward > 90.0
    assert terminated is True
    assert truncated is False
    assert info["reached_goal"] is False
    assert step_info["reached_goal"] is True
    assert step_info["collided"] is False
    assert step_info["timed_out"] is False
    assert step_info["episode_metrics"].success is True
    assert step_info["reward_breakdown"].arrive == pytest.approx(100.0)
    assert step_info["raw_reward"] == -999.0


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


class FakeGoalVelocityAviary:
    def reset(self, seed=None, options=None):
        return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]], {
            "seed": seed
        }

    def step(self, action):
        return (
            [[0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]],
            -999.0,
            False,
            False,
            {"source": "fake-goal"},
        )

    def close(self):
        pass
