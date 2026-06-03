from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swift.config import SimulationSettings
from swift.core import DroneAction, ObstacleState


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
        enable_obstacles: bool = False,
        swift_obstacles: Sequence[ObstacleState] | None = None,
        velocity_aviary_cls: Any | None = None,
        drone_model: Any | None = None,
        physics: Any | None = None,
    ) -> None:
        self.settings = settings
        self.max_speed = float(max_speed)
        self.enable_obstacles = bool(enable_obstacles)
        self._swift_obstacles = tuple(swift_obstacles or ())
        self._last_observation: tuple[float, ...] | None = None
        self._obstacle_body_ids: tuple[int, ...] = ()
        self._swift_obstacle_body_ids: tuple[int, ...] = ()
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
            obstacles=self.enable_obstacles,
            user_debug_gui=False,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        raw_observation, raw_info = self._env.reset(seed=seed, options=options)
        self._swift_obstacle_body_ids = ()
        self._obstacle_body_ids = ()
        observation = pybullet_observation_to_swift(raw_observation)
        self._last_observation = observation
        return observation, self._info(raw_info, self._contact_info(observation))

    def step(self, action: DroneAction) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        command = drone_action_to_velocity_command(action, self.max_speed)
        raw_observation, reward, terminated, truncated, raw_info = self._env.step(_action_array(command))
        observation = pybullet_observation_to_swift(raw_observation)
        self._last_observation = observation
        return observation, float(reward), bool(terminated), bool(truncated), self._info(
            raw_info,
            self._contact_info(observation),
        )

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

    def _contact_info(self, observation: tuple[float, ...]) -> dict[str, Any]:
        if not self.enable_obstacles and not self._swift_obstacles:
            return {}
        try:
            import pybullet as p
        except ImportError:
            return {}

        client_getter = getattr(self._env, "getPyBulletClient", None)
        drone_getter = getattr(self._env, "getDroneIds", None)
        if client_getter is None or drone_getter is None:
            return {}
        client = client_getter()
        drone_ids = tuple(int(drone_id) for drone_id in drone_getter())
        self._ensure_swift_obstacles(p, client)
        obstacle_ids = self._obstacle_ids(p, client, drone_ids)
        if not drone_ids or not obstacle_ids:
            return {}

        contacts = []
        closest_distances: list[tuple[float, int]] = []
        for drone_id in drone_ids:
            for obstacle_id in obstacle_ids:
                contacts.extend(
                    p.getContactPoints(bodyA=drone_id, bodyB=obstacle_id, physicsClientId=client)
                )
                for point in p.getClosestPoints(
                    bodyA=drone_id,
                    bodyB=obstacle_id,
                    distance=1_000_000.0,
                    physicsClientId=client,
                ):
                    closest_distances.append((float(point[8]), obstacle_id))

        nearest_distance = None
        nearest_obstacle_id = None
        if closest_distances:
            nearest_distance, nearest_obstacle_id = min(closest_distances, key=lambda item: item[0])
        elif contacts:
            nearest_distance = min(float(point[8]) for point in contacts)
            nearest_obstacle_id = int(contacts[0][2])

        info: dict[str, Any] = {
            "contact_count": len(contacts),
            "collided": bool(contacts),
        }
        if nearest_distance is not None:
            info["minimum_safety_distance"] = float(nearest_distance)
        if nearest_obstacle_id is not None:
            info["nearest_obstacle_body_id"] = int(nearest_obstacle_id)
            relative = self._relative_body_position(p, client, nearest_obstacle_id, observation[0:3])
            if relative is not None:
                info["nearest_obstacle_relative"] = relative
            radius = self._body_radius(p, client, nearest_obstacle_id)
            if radius is not None:
                info["nearest_obstacle_radius"] = radius
        return info

    def _obstacle_ids(self, pybullet_module: Any, client: Any, drone_ids: tuple[int, ...]) -> tuple[int, ...]:
        if self._swift_obstacle_body_ids:
            self._obstacle_body_ids = self._swift_obstacle_body_ids
            return self._swift_obstacle_body_ids
        total_bodies = int(pybullet_module.getNumBodies(physicsClientId=client))
        plane_id = getattr(self._env, "PLANE_ID", None)
        excluded = set(drone_ids)
        if plane_id is not None:
            excluded.add(int(plane_id))
        obstacle_ids = tuple(body_id for body_id in range(total_bodies) if body_id not in excluded)
        self._obstacle_body_ids = obstacle_ids
        return obstacle_ids

    def _ensure_swift_obstacles(self, pybullet_module: Any, client: Any) -> None:
        if not self._swift_obstacles or self._swift_obstacle_body_ids:
            return
        create_shape = getattr(pybullet_module, "createCollisionShape", None)
        create_body = getattr(pybullet_module, "createMultiBody", None)
        if create_shape is None or create_body is None:
            return
        body_ids: list[int] = []
        for obstacle in self._swift_obstacles:
            try:
                shape_id = create_shape(
                    shapeType=getattr(pybullet_module, "GEOM_SPHERE", 2),
                    radius=float(obstacle.radius),
                    physicsClientId=client,
                )
                body_id = create_body(
                    baseMass=0.0,
                    baseCollisionShapeIndex=shape_id,
                    basePosition=tuple(float(value) for value in obstacle.position),
                    physicsClientId=client,
                )
            except Exception:
                continue
            body_ids.append(int(body_id))
        self._swift_obstacle_body_ids = tuple(body_ids)

    @staticmethod
    def _relative_body_position(
        pybullet_module: Any,
        client: Any,
        body_id: int,
        drone_position: tuple[float, ...],
    ) -> tuple[float, float, float] | None:
        getter = getattr(pybullet_module, "getBasePositionAndOrientation", None)
        if getter is None:
            return None
        position, _ = getter(bodyUniqueId=int(body_id), physicsClientId=client)
        if position is None or len(position) != 3:
            return None
        return tuple(float(position[index]) - float(drone_position[index]) for index in range(3))

    @staticmethod
    def _body_radius(pybullet_module: Any, client: Any, body_id: int) -> float | None:
        getter = getattr(pybullet_module, "getAABB", None)
        if getter is None:
            return None
        try:
            bounds = getter(bodyUniqueId=int(body_id), physicsClientId=client)
        except Exception:
            return None
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            return None
        lower, upper = bounds
        if lower is None or upper is None or len(lower) != 3 or len(upper) != 3:
            return None
        radius = max((float(upper[index]) - float(lower[index])) / 2.0 for index in range(3))
        if not math.isfinite(radius) or radius < 0.0:
            return None
        return radius

    @staticmethod
    def _info(raw_info: Any, runtime_info: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = raw_info if isinstance(raw_info, dict) else {"value": raw_info}
        info: dict[str, Any] = {
            "backend": "pybullet_velocity_aviary",
            "runtime_contract": "smoke",
            "raw_info": raw,
        }
        for source_key, target_key in (
            ("collided", "collided"),
            ("collision", "collided"),
            ("timed_out", "timed_out"),
            ("minimum_safety_distance", "minimum_safety_distance"),
            ("nearest_obstacle_radius", "nearest_obstacle_radius"),
            ("obstacle_radius", "nearest_obstacle_radius"),
        ):
            if source_key in raw:
                info[target_key] = raw[source_key]
        for source_key in ("nearest_obstacle_relative", "relative_obstacle", "obstacle_relative"):
            if source_key in raw:
                info["nearest_obstacle_relative"] = raw[source_key]
                break
        info.update(runtime_info or {})
        return {
            **info,
            "collided": bool(info.get("collided", False)),
            "timed_out": bool(info.get("timed_out", False)),
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
