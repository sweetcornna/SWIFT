import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import swift.sim.pybullet_runtime as pybullet_runtime
from swift.config import SimulationSettings
from swift.core import DroneAction, ObstacleState
from swift.envs import PyBulletVelocityTrainingEnv, SimpleAvoidanceSettings
from swift.sim.pybullet_runtime import (
    PyBulletRuntimeUnavailableError,
    PyBulletVelocityRuntimeEnv,
    _pixi_windows_runtime_directories,
    _prepare_windows_pixi_runtime,
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


def test_pixi_windows_runtime_directories_include_dll_search_paths(tmp_path: Path):
    pixi_env = tmp_path / ".pixi" / "envs" / "default"
    expected = (pixi_env, pixi_env / "Library" / "bin", pixi_env / "Scripts")
    for path in expected:
        path.mkdir(parents=True)

    assert _pixi_windows_runtime_directories(make_settings(tmp_path)) == expected


def test_prepare_windows_pixi_runtime_prepends_existing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pixi_env = tmp_path / ".pixi" / "envs" / "default"
    expected = (pixi_env, pixi_env / "Library" / "bin", pixi_env / "Scripts")
    for path in expected:
        path.mkdir(parents=True)
    monkeypatch.setenv("PATH", r"C:\existing")

    _prepare_windows_pixi_runtime(make_settings(tmp_path))

    raw_parts = tuple(Path(part) for part in os.environ["PATH"].split(";") if part)
    assert raw_parts[:3] == expected


def test_prepare_windows_pixi_runtime_retains_dll_directory_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pixi_env = tmp_path / ".pixi" / "envs" / "default"
    expected = (pixi_env, pixi_env / "Library" / "bin", pixi_env / "Scripts")
    for path in expected:
        path.mkdir(parents=True)
    handles = [object(), object(), object()]
    registered = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PATH", r"C:\existing")
    pybullet_runtime._DLL_DIRECTORY_HANDLES.clear()

    def fake_add_dll_directory(path):
        registered.append(Path(path))
        return handles[len(registered) - 1]

    monkeypatch.setattr(os, "add_dll_directory", fake_add_dll_directory)

    _prepare_windows_pixi_runtime(make_settings(tmp_path))

    assert tuple(registered) == expected
    assert pybullet_runtime._DLL_DIRECTORY_HANDLES == handles


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


def test_runtime_env_can_enable_pybullet_obstacles(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    captured_kwargs = {}

    def aviary_factory(**kwargs):
        captured_kwargs["kwargs"] = kwargs
        return FakeVelocityAviary()

    runtime = PyBulletVelocityRuntimeEnv(
        make_settings(tmp_path),
        enable_obstacles=True,
        velocity_aviary_cls=aviary_factory,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )
    runtime.close()

    assert captured_kwargs["kwargs"]["obstacles"] is True


def test_runtime_env_detects_headless_pybullet_contacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_pybullet = FakePyBulletContacts()
    monkeypatch.setitem(sys.modules, "pybullet", fake_pybullet)

    runtime = PyBulletVelocityRuntimeEnv(
        make_settings(tmp_path),
        enable_obstacles=True,
        velocity_aviary_cls=lambda **_: FakeContactVelocityAviary(),
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    _, reset_info = runtime.reset(seed=29)
    _, _, _, _, step_info = runtime.step(DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0))
    runtime.close()

    assert reset_info["contact_count"] == 1
    assert reset_info["collided"] is True
    assert reset_info["minimum_safety_distance"] == pytest.approx(-0.025)
    assert reset_info["nearest_obstacle_body_id"] == 2
    assert reset_info["nearest_obstacle_radius"] == pytest.approx(0.2)
    assert fake_pybullet.closest_query_distances
    assert max(fake_pybullet.closest_query_distances) == pytest.approx(10.0)
    assert step_info["contact_count"] == 1
    assert step_info["collided"] is True
    assert step_info["minimum_safety_distance"] == pytest.approx(-0.025)
    assert step_info["nearest_obstacle_radius"] == pytest.approx(0.2)


def test_runtime_env_omits_obstacle_radius_when_aabb_lookup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_pybullet = FakePyBulletContactsWithBrokenAabb()
    monkeypatch.setitem(sys.modules, "pybullet", fake_pybullet)

    runtime = PyBulletVelocityRuntimeEnv(
        make_settings(tmp_path),
        enable_obstacles=True,
        velocity_aviary_cls=lambda **_: FakeContactVelocityAviary(),
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    _, reset_info = runtime.reset(seed=37)
    _, _, _, _, step_info = runtime.step(DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0))
    runtime.close()

    assert reset_info["nearest_obstacle_relative"] == pytest.approx((0.5, 0.5, 0.5))
    assert "nearest_obstacle_radius" not in reset_info
    assert "nearest_obstacle_radius" not in step_info


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
    assert step_one[4]["runtime_contract"] == "pybullet_velocity_training_compatibility"
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


def test_pybullet_training_env_normalizes_reward_for_high_frequency_steps(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_env = FakeSmallProgressVelocityAviary()
    settings = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.0),
        goal=(2.0, 0.0, 0.0),
        goal_radius=0.2,
        max_steps=500,
        max_speed=1.0,
    )
    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        velocity_aviary_cls=lambda **_: fake_env,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    training_env.reset(seed=15)
    _, reward, terminated, truncated, step_info = training_env.step(
        DroneAction(speed=1.0, heading_delta=0.0, climb_rate=0.0)
    )

    assert terminated is False
    assert truncated is False
    assert step_info["reward_breakdown"].approach == pytest.approx(0.05 / 2.0)
    assert step_info["reward_breakdown"].timeliness == pytest.approx(-1.0 / settings.max_steps)
    assert reward == pytest.approx((0.05 / 2.0) - (1.0 / settings.max_steps))


def test_pybullet_training_env_propagates_runtime_collision_and_obstacle_metrics(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_env = FakeCollisionVelocityAviary()
    settings = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 0.0, 0.0),
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

    observation, _ = training_env.reset(seed=17)
    next_observation, reward, terminated, truncated, step_info = training_env.step(
        DroneAction(speed=1.0, heading_delta=0.0, climb_rate=0.0)
    )

    assert observation[13] == pytest.approx(0.0)
    assert next_observation[10:13] == pytest.approx((0.0, 0.0, 0.0))
    assert next_observation[13] == pytest.approx(0.2)
    assert reward < -90.0
    assert terminated is True
    assert truncated is False
    assert step_info["reached_goal"] is False
    assert step_info["collided"] is True
    assert step_info["timed_out"] is False
    assert step_info["reward_breakdown"].obstacle == pytest.approx(-100.0)
    assert step_info["minimum_safety_distance"] == pytest.approx(-0.05)
    assert step_info["episode_metrics"].success is False
    assert step_info["episode_metrics"].collided is True
    assert step_info["episode_metrics"].minimum_safety_distance == pytest.approx(-0.05)


def test_pybullet_training_env_does_not_enable_builtin_obstacles_without_configured_swift_obstacles(
    tmp_path: Path,
):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    captured_kwargs = {}
    settings = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 0.0, 0.0),
        obstacles=(),
        max_steps=4,
        safety_margin=0.1,
    )

    def aviary_factory(**kwargs):
        captured_kwargs["kwargs"] = kwargs
        return FakeVelocityAviary()

    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        enable_pybullet_obstacles=True,
        velocity_aviary_cls=aviary_factory,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    observation, reset_info = training_env.reset(seed=31)

    assert captured_kwargs["kwargs"]["obstacles"] is False
    assert training_env._runtime._swift_obstacles == ()
    assert observation[10:13] == pytest.approx((0.0, 0.0, 0.0))
    assert observation[13] == pytest.approx(0.0)
    assert reset_info["minimum_safety_distance"] == pytest.approx(0.0)


def test_pybullet_training_env_uses_swift_obstacle_tail_without_builtin_pybullet_obstacles(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    captured_kwargs = {}
    settings = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 0.0, 0.0),
        obstacles=(ObstacleState(position=(1.0, 0.0, 0.0), radius=0.25),),
        max_steps=4,
        safety_margin=0.1,
    )

    def aviary_factory(**kwargs):
        captured_kwargs["kwargs"] = kwargs
        return FakeVelocityAviary()

    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        velocity_aviary_cls=aviary_factory,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    observation, reset_info = training_env.reset(seed=19)
    next_observation, _, terminated, _, step_info = training_env.step(
        DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
    )

    assert captured_kwargs["kwargs"]["obstacles"] is False
    assert observation[10:13] == pytest.approx((0.0, -2.0, -3.0))
    assert observation[13] == pytest.approx(0.25)
    assert reset_info["minimum_safety_distance"] == pytest.approx(math.sqrt(13.0) - 0.25)
    assert next_observation[13] == pytest.approx(0.25)
    assert terminated is False
    assert step_info["collided"] is False


def test_runtime_env_injects_and_tracks_swift_configured_obstacle_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_pybullet = FakePyBulletSwiftObstacles()
    monkeypatch.setitem(sys.modules, "pybullet", fake_pybullet)

    runtime = PyBulletVelocityRuntimeEnv(
        make_settings(tmp_path),
        enable_obstacles=True,
        swift_obstacles=(ObstacleState(position=(2.0, 3.0, 4.0), radius=0.4),),
        velocity_aviary_cls=lambda **_: FakeContactVelocityAviary(),
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    observation, reset_info = runtime.reset(seed=41)
    runtime.close()

    assert observation[0:3] == pytest.approx((1.0, 2.0, 3.0))
    assert fake_pybullet.collision_shapes == [(0.4, 123)]
    assert fake_pybullet.multi_bodies == [(900, (2.0, 3.0, 4.0), 123)]
    assert fake_pybullet.contact_queries == [(1, 900, 123)]
    assert fake_pybullet.closest_query_distances
    assert max(fake_pybullet.closest_query_distances) == pytest.approx(10.0)
    assert reset_info["nearest_obstacle_body_id"] == 900
    assert reset_info["nearest_obstacle_relative"] == pytest.approx((1.0, 1.0, 1.0))
    assert reset_info["nearest_obstacle_radius"] == pytest.approx(0.4)


def test_runtime_env_recreates_swift_obstacle_bodies_after_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_pybullet = FakePyBulletSwiftObstacles()
    monkeypatch.setitem(sys.modules, "pybullet", fake_pybullet)
    runtime = PyBulletVelocityRuntimeEnv(
        make_settings(tmp_path),
        enable_obstacles=True,
        swift_obstacles=(ObstacleState(position=(2.0, 3.0, 4.0), radius=0.4),),
        velocity_aviary_cls=lambda **_: FakeContactVelocityAviary(),
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    runtime.reset(seed=43)
    runtime.reset(seed=44)
    runtime.close()

    assert fake_pybullet.multi_bodies == [
        (900, (2.0, 3.0, 4.0), 123),
        (901, (2.0, 3.0, 4.0), 123),
    ]
    assert fake_pybullet.contact_queries[-1] == (1, 901, 123)
    assert runtime._swift_obstacle_body_ids == (901,)


def test_pybullet_training_env_passes_configured_obstacles_to_runtime_when_enabled(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    captured_kwargs = {}
    settings = SimpleAvoidanceSettings(
        obstacles=(ObstacleState(position=(2.0, 0.0, 1.0), radius=0.3),),
    )

    def aviary_factory(**kwargs):
        captured_kwargs["kwargs"] = kwargs
        return FakeVelocityAviary()

    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        enable_pybullet_obstacles=True,
        velocity_aviary_cls=aviary_factory,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    training_env.close()

    assert captured_kwargs["kwargs"]["obstacles"] is False
    assert training_env._runtime._swift_obstacles == settings.obstacles


def test_pybullet_training_env_injects_configured_obstacles_even_when_builtin_obstacles_are_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    fake_pybullet = FakePyBulletSwiftObstacles()
    monkeypatch.setitem(sys.modules, "pybullet", fake_pybullet)
    captured_kwargs = {}
    settings = SimpleAvoidanceSettings(
        obstacles=(ObstacleState(position=(2.0, 3.0, 4.0), radius=0.4),),
    )

    def aviary_factory(**kwargs):
        captured_kwargs["kwargs"] = kwargs
        return FakeContactVelocityAviary()

    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        enable_pybullet_obstacles=True,
        velocity_aviary_cls=aviary_factory,
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    observation, info = training_env.reset(seed=47)
    training_env.close()

    assert captured_kwargs["kwargs"]["obstacles"] is False
    assert fake_pybullet.multi_bodies == [(900, (2.0, 3.0, 4.0), 123)]
    assert observation[10:13] == pytest.approx((1.0, 1.0, 1.0))
    assert observation[13] == pytest.approx(0.4)
    assert info["nearest_obstacle_body_id"] == 900


def test_pybullet_training_env_detects_collision_against_configured_static_obstacle(tmp_path: Path):
    vendored = tmp_path / "external" / "gym-pybullet-drones"
    vendored.mkdir(parents=True)
    settings = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 0.0, 0.0),
        obstacles=(ObstacleState(position=(1.1, 2.2, 3.3), radius=0.3),),
        max_steps=4,
        safety_margin=0.1,
    )
    training_env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=settings,
        velocity_aviary_cls=lambda **_: FakeVelocityAviary(),
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    training_env.reset(seed=23)
    next_observation, reward, terminated, truncated, step_info = training_env.step(
        DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0)
    )

    assert next_observation[10:13] == pytest.approx((0.0, 0.0, 0.0))
    assert next_observation[13] == pytest.approx(0.3)
    assert reward < -90.0
    assert terminated is True
    assert truncated is False
    assert step_info["collided"] is True
    assert step_info["episode_metrics"].success is False
    assert step_info["episode_metrics"].minimum_safety_distance == pytest.approx(-0.3)


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


class FakeSmallProgressVelocityAviary:
    def reset(self, seed=None, options=None):
        return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0, 0, 0, 0]], {
            "seed": seed
        }

    def step(self, action):
        return (
            [[0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.05, 0.0, 0.0, 0, 0, 0, 0]],
            -999.0,
            False,
            False,
            {"source": "fake-small-progress"},
        )

    def close(self):
        pass


class FakeCollisionVelocityAviary:
    def reset(self, seed=None, options=None):
        return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]], {
            "seed": seed
        }

    def step(self, action):
        return (
            [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]],
            -7.0,
            False,
            False,
            {
                "collided": True,
                "minimum_safety_distance": -0.05,
                "nearest_obstacle_relative": (0.0, 0.0, 0.0),
                "nearest_obstacle_radius": 0.2,
            },
        )

    def close(self):
        pass


class FakeContactVelocityAviary(FakeVelocityAviary):
    PLANE_ID = 0

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        info.update({"collided": False, "minimum_safety_distance": 999.0})
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update({"collided": False, "minimum_safety_distance": 999.0})
        return observation, reward, terminated, truncated, info

    def getPyBulletClient(self):
        return 123

    def getDroneIds(self):
        return [1]


class FakePyBulletContacts:
    def __init__(self) -> None:
        self.closest_query_distances = []

    def getNumBodies(self, physicsClientId=None):
        assert physicsClientId == 123
        return 3

    def getContactPoints(self, bodyA=None, bodyB=None, physicsClientId=None):
        assert bodyA == 1
        assert bodyB == 2
        assert physicsClientId == 123
        return [(0, bodyA, bodyB, -1, -1, (0, 0, 0), (0, 0, 0), (0, 0, 1), -0.025, 3.0)]

    def getClosestPoints(self, bodyA=None, bodyB=None, distance=None, physicsClientId=None):
        assert bodyA == 1
        assert bodyB == 2
        assert physicsClientId == 123
        self.closest_query_distances.append(float(distance))
        return [(0, bodyA, bodyB, -1, -1, (0, 0, 0), (0, 0, 0), (0, 0, 1), -0.025, 3.0)]

    def getBasePositionAndOrientation(self, bodyUniqueId=None, physicsClientId=None):
        assert bodyUniqueId == 2
        assert physicsClientId == 123
        return (1.5, 2.5, 3.5), (0, 0, 0, 1)

    def getAABB(self, bodyUniqueId=None, physicsClientId=None):
        assert bodyUniqueId == 2
        assert physicsClientId == 123
        return (1.3, 2.3, 3.3), (1.7, 2.7, 3.7)


class FakePyBulletContactsWithBrokenAabb(FakePyBulletContacts):
    def getAABB(self, bodyUniqueId=None, physicsClientId=None):
        raise RuntimeError("AABB unavailable")


class FakePyBulletSwiftObstacles(FakePyBulletContacts):
    GEOM_SPHERE = 2

    def __init__(self) -> None:
        self.closest_query_distances = []
        self.collision_shapes = []
        self.multi_bodies = []
        self.contact_queries = []
        self._next_body_id = 900

    def getNumBodies(self, physicsClientId=None):
        raise AssertionError("tracked SWIFT obstacle bodies should avoid scanning all PyBullet bodies")

    def createCollisionShape(self, shapeType=None, radius=None, physicsClientId=None):
        assert shapeType == self.GEOM_SPHERE
        self.collision_shapes.append((radius, physicsClientId))
        return self._next_body_id

    def createMultiBody(
        self,
        baseMass=None,
        baseCollisionShapeIndex=None,
        basePosition=None,
        physicsClientId=None,
    ):
        assert baseMass == 0.0
        self.multi_bodies.append((baseCollisionShapeIndex, tuple(basePosition), physicsClientId))
        body_id = self._next_body_id
        self._next_body_id += 1
        return body_id

    def getContactPoints(self, bodyA=None, bodyB=None, physicsClientId=None):
        self.contact_queries.append((bodyA, bodyB, physicsClientId))
        return []

    def getClosestPoints(self, bodyA=None, bodyB=None, distance=None, physicsClientId=None):
        assert bodyA == 1
        assert bodyB in {900, 901}
        assert physicsClientId == 123
        self.closest_query_distances.append(float(distance))
        return [(0, bodyA, bodyB, -1, -1, (0, 0, 0), (0, 0, 0), (0, 0, 1), 0.6, 3.0)]

    def getBasePositionAndOrientation(self, bodyUniqueId=None, physicsClientId=None):
        assert bodyUniqueId in {900, 901}
        assert physicsClientId == 123
        return (2.0, 3.0, 4.0), (0, 0, 0, 1)

    def getAABB(self, bodyUniqueId=None, physicsClientId=None):
        assert bodyUniqueId in {900, 901}
        assert physicsClientId == 123
        return (1.6, 2.6, 3.6), (2.4, 3.4, 4.4)
