from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from swift.core import (
    DroneAction,
    EpisodeMetrics,
    RewardBreakdown,
    Vector3,
    compute_path_length,
    compute_path_smoothness,
)
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
        self._path: list[Vector3] = []
        self._previous_goal_distance = 0.0
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
        observation = self._with_goal_tail(observation)
        self._path = [observation[0:3]]
        self._previous_goal_distance = float(observation[14])
        return observation, self._info(
            info,
            reward_breakdown=RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
            reached_goal=False,
            timed_out=False,
        )

    def step(
        self,
        action: DroneAction | Sequence[float],
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        drone_action = _coerce_action(action)
        observation, raw_reward, raw_terminated, raw_truncated, info = self._runtime.step(drone_action)
        observation = self._with_goal_tail(observation)
        self._steps += 1
        current_goal_distance = float(observation[14])
        reached_goal = current_goal_distance <= self.settings.goal_radius
        terminated = bool(raw_terminated or reached_goal)
        timed_out = (not terminated and not raw_truncated and self._steps >= self.settings.max_steps) or bool(
            info.get("timed_out", False)
        )
        truncated = bool(raw_truncated or timed_out)
        self._path.append(observation[0:3])
        reward_breakdown = RewardBreakdown(
            arrive=100.0 if reached_goal else 0.0,
            approach=self._previous_goal_distance - current_goal_distance,
            obstacle=0.0,
            smoothness=-0.05 * abs(float(drone_action.heading_delta)),
            timeliness=-1.0,
        )
        self._previous_goal_distance = current_goal_distance
        return (
            observation,
            reward_breakdown.total,
            terminated,
            truncated,
            self._info(
                info,
                reward_breakdown=reward_breakdown,
                reached_goal=reached_goal,
                timed_out=timed_out,
                raw_reward=float(raw_reward),
                episode_done=terminated or truncated,
            ),
        )

    def close(self) -> None:
        self._runtime.close()

    def _with_goal_tail(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        position = observation[0:3]
        relative_goal = tuple(self.settings.goal[index] - position[index] for index in range(3))
        goal_distance = _distance(position, self.settings.goal)
        return (
            *observation[0:7],
            *relative_goal,
            0.0,
            0.0,
            0.0,
            0.0,
            goal_distance,
        )

    def _info(
        self,
        runtime_info: dict[str, Any],
        *,
        reward_breakdown: RewardBreakdown,
        reached_goal: bool,
        timed_out: bool,
        raw_reward: float | None = None,
        episode_done: bool = False,
    ) -> dict[str, Any]:
        info = dict(runtime_info)
        info.update(
            {
                "runtime_contract": "goal_only_training_compatibility",
                "steps": self._steps,
                "reached_goal": bool(reached_goal),
                "collided": False,
                "timed_out": bool(timed_out),
                "reward_breakdown": reward_breakdown,
            }
        )
        if raw_reward is not None:
            info["raw_reward"] = float(raw_reward)
        if episode_done:
            info["episode_metrics"] = EpisodeMetrics(
                reached_goal=bool(reached_goal),
                collided=False,
                timed_out=bool(timed_out),
                path_length=compute_path_length(tuple(self._path)),
                path_smoothness=compute_path_smoothness(tuple(self._path)),
                minimum_safety_distance=0.0,
                steps=self._steps,
            )
        return info


def _coerce_action(action: DroneAction | Sequence[float]) -> DroneAction:
    if isinstance(action, DroneAction):
        return action
    if len(action) != 3:
        raise ValueError("action must contain exactly 3 values")
    return DroneAction(speed=float(action[0]), heading_delta=float(action[1]), climb_rate=float(action[2]))


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(first[index]) - float(second[index])) ** 2 for index in range(3)))
