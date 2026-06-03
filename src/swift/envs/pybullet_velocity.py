from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from swift.core import DroneAction
from swift.envs.simple_avoidance import SimpleAvoidanceSettings

if TYPE_CHECKING:
    from swift.config import SimulationSettings
    from swift.sim.pybullet_runtime import PyBulletVelocityRuntimeEnv


class PyBulletVelocityTrainingEnv:
    observation_shape = (15,)
    action_shape = (3,)

    def __init__(
        self,
        *,
        simulation_settings: SimulationSettings,
        settings: SimpleAvoidanceSettings | None = None,
        runtime: PyBulletVelocityRuntimeEnv | None = None,
        velocity_aviary_cls: Any | None = None,
        drone_model: Any | None = None,
        physics: Any | None = None,
    ) -> None:
        self.settings = settings or SimpleAvoidanceSettings()
        self._steps = 0
        if runtime is None:
            from swift.sim.pybullet_runtime import PyBulletVelocityRuntimeEnv

            runtime = PyBulletVelocityRuntimeEnv(
                simulation_settings,
                max_speed=self.settings.max_speed,
                velocity_aviary_cls=velocity_aviary_cls,
                drone_model=drone_model,
                physics=physics,
            )
        self._runtime = runtime

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        self._steps = 0
        observation, info = self._runtime.reset(seed=seed, options=options)
        return observation, self._info(info, timed_out=False)

    def step(
        self,
        action: DroneAction | Sequence[float],
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._runtime.step(_coerce_action(action))
        self._steps += 1
        timed_out = (not terminated and not truncated and self._steps >= self.settings.max_steps) or bool(
            info.get("timed_out", False)
        )
        truncated = bool(truncated or timed_out)
        return observation, reward, bool(terminated), truncated, self._info(info, timed_out=timed_out)

    def close(self) -> None:
        self._runtime.close()

    def _info(self, runtime_info: dict[str, Any], *, timed_out: bool) -> dict[str, Any]:
        info = dict(runtime_info)
        info.update(
            {
                "runtime_contract": "training_compatibility",
                "steps": self._steps,
                "reached_goal": False,
                "collided": False,
                "timed_out": bool(timed_out),
            }
        )
        return info


def _coerce_action(action: DroneAction | Sequence[float]) -> DroneAction:
    if isinstance(action, DroneAction):
        return action
    if len(action) != 3:
        raise ValueError("action must contain exactly 3 values")
    return DroneAction(speed=float(action[0]), heading_delta=float(action[1]), climb_rate=float(action[2]))
