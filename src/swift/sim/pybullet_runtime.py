from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from swift.config import SimulationSettings
from swift.core import DroneAction


class PyBulletRuntimeUnavailableError(RuntimeError):
    """Raised when the optional PyBullet runtime substrate is unavailable."""


def build_pybullet_drones_path(settings: SimulationSettings) -> Path:
    return settings.pybullet_root / "external" / "gym-pybullet-drones"


def drone_action_to_velocity_command(action: DroneAction, max_speed: float) -> list[list[float]]:
    speed = _clamp(float(action.speed), 0.0, max(0.0, float(max_speed)))
    if speed == 0.0:
        return [[0.0, 0.0, 0.0, 0.0]]
    speed_fraction = 0.0 if max_speed <= 0.0 else _clamp(speed / float(max_speed), 0.0, 1.0)
    climb_component = _clamp(float(action.climb_rate) / speed, -1.0, 1.0)
    heading = float(action.heading_delta)
    return [[math.cos(heading), math.sin(heading), climb_component, speed_fraction]]


class PyBulletVelocityRuntimeEnv:
    observation_shape = (15,)
    action_shape = (3,)

    def __init__(
        self,
        settings: SimulationSettings,
        *,
        max_speed: float = 1.0,
        velocity_aviary_cls: Any | None = None,
        drone_model: Any | None = None,
        physics: Any | None = None,
    ) -> None:
        self.settings = settings
        self.max_speed = float(max_speed)
        self._last_observation: tuple[float, ...] | None = None
        aviary_cls, drone_model_value, physics_value = self._resolve_runtime(
            settings=settings,
            velocity_aviary_cls=velocity_aviary_cls,
            drone_model=drone_model,
            physics=physics,
        )
        self._env = aviary_cls(
            drone_model=drone_model_value.CF2X,
            num_drones=1,
            physics=physics_value.PYB,
            gui=False,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        raw_observation, raw_info = self._env.reset(seed=seed, options=options)
        observation = pybullet_observation_to_swift(raw_observation)
        self._last_observation = observation
        return observation, self._info(raw_info)

    def step(self, action: DroneAction) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        command = drone_action_to_velocity_command(action, self.max_speed)
        raw_observation, reward, terminated, truncated, raw_info = self._env.step(_action_array(command))
        observation = pybullet_observation_to_swift(raw_observation)
        self._last_observation = observation
        return observation, float(reward), bool(terminated), bool(truncated), self._info(raw_info)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()

    @staticmethod
    def _resolve_runtime(
        *,
        settings: SimulationSettings,
        velocity_aviary_cls: Any | None,
        drone_model: Any | None,
        physics: Any | None,
    ) -> tuple[Any, Any, Any]:
        if velocity_aviary_cls is not None and drone_model is not None and physics is not None:
            return velocity_aviary_cls, drone_model, physics

        vendored_path = build_pybullet_drones_path(settings)
        if not vendored_path.is_dir():
            raise PyBulletRuntimeUnavailableError(f"gym-pybullet-drones path not found: {vendored_path}")
        if str(vendored_path) not in sys.path:
            sys.path.insert(0, str(vendored_path))

        try:
            from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
            from gym_pybullet_drones.utils.enums import DroneModel, Physics
        except ImportError as exc:
            raise PyBulletRuntimeUnavailableError(f"PyBullet runtime imports failed: {exc}") from exc
        return VelocityAviary, DroneModel, Physics

    @staticmethod
    def _info(raw_info: Any) -> dict[str, Any]:
        return {
            "backend": "pybullet_velocity_aviary",
            "runtime_contract": "smoke",
            "raw_info": raw_info if isinstance(raw_info, dict) else {"value": raw_info},
        }


def pybullet_observation_to_swift(raw_observation: Any) -> tuple[float, ...]:
    row = _first_drone_observation(raw_observation)
    if len(row) < 13:
        raise ValueError("PyBullet observation must contain at least 13 values")
    position = tuple(float(value) for value in row[0:3])
    yaw = float(row[9])
    velocity = tuple(float(value) for value in row[10:13])
    return (
        *position,
        *velocity,
        yaw,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _first_drone_observation(raw_observation: Any) -> list[float]:
    rows = raw_observation.tolist() if hasattr(raw_observation, "tolist") else raw_observation
    if not rows:
        raise ValueError("PyBullet observation must not be empty")
    row = rows[0] if isinstance(rows[0], (list, tuple)) else rows
    return [float(value) for value in row]


def _action_array(command: list[list[float]]) -> Any:
    try:
        import numpy as np
    except ImportError:
        return command
    return np.array(command, dtype=np.float32)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
