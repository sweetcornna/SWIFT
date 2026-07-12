import math
from pathlib import Path

import pytest

from swift.config import PyBulletObstacleRandomizationSettings, SimulationSettings
from swift.envs import PyBulletVelocityTrainingEnv, SimpleAvoidanceSettings


def test_randomized_pybullet_observation_appends_sorted_obstacle_slots(tmp_path: Path) -> None:
    settings = SimpleAvoidanceSettings(
        start=(0.0, 0.0, 0.1125),
        goal=(0.5, 0.0, 0.1125),
        max_steps=4,
        max_speed=1.0,
        max_climb_rate=0.0,
        safety_margin=0.1,
        obstacles=(),
    )
    randomization = PyBulletObstacleRandomizationSettings(
        enabled=True,
        min_obstacles=3,
        max_obstacles=3,
        radius_range=(0.04, 0.08),
        x_range=(0.12, 0.38),
        y_range=(-0.3, 0.3),
        z=0.1125,
        require_path_blocker=True,
    )
    runtime = FakeRuntime(position=settings.start)
    env = PyBulletVelocityTrainingEnv(
        simulation_settings=_simulation_settings(tmp_path),
        settings=settings,
        runtime=runtime,
        enable_pybullet_obstacles=True,
        obstacle_randomization=randomization,
    )

    observation, info = env.reset(seed=1234)

    assert env.observation_shape == (27,)
    assert len(observation) == 27
    expected = sorted(info["obstacles"], key=lambda obstacle: math.dist(settings.start, obstacle.position))
    expected_slots = []
    for obstacle in expected:
        expected_slots.extend(
            [
                obstacle.position[0] - settings.start[0],
                obstacle.position[1] - settings.start[1],
                obstacle.position[2] - settings.start[2],
                obstacle.radius,
            ]
        )
    assert observation[15:] == pytest.approx(expected_slots)
    assert observation[10:14] == pytest.approx(expected_slots[:4])
    assert sorted(runtime.obstacles, key=lambda obstacle: math.dist(settings.start, obstacle.position)) == expected


class FakeRuntime:
    def __init__(self, *, position: tuple[float, float, float]) -> None:
        self.position = position
        self.obstacles = ()

    def set_swift_obstacles(self, obstacles):
        self.obstacles = tuple(obstacles)

    def reset(self, seed=None, options=None):
        return (
            *self.position,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ), {"seed": seed}

    def step(self, action):
        observation, info = self.reset()
        return observation, 0.0, False, False, info

    def close(self):
        pass


def _simulation_settings(tmp_path: Path) -> SimulationSettings:
    return SimulationSettings(
        pybullet_root=tmp_path,
        pixi_executable=tmp_path / "pixi.exe",
        required_tasks=(),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )
