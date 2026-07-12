from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
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
from swift.envs.pybullet_obstacle_randomization import sample_pybullet_obstacles

if TYPE_CHECKING:
    from swift.config import (
        PyBulletCurriculumPhaseSettings,
        PyBulletCurriculumSettings,
        PyBulletObstacleRandomizationSettings,
        PyBulletRewardSettings,
        SimulationSettings,
    )
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
        enable_pybullet_obstacles: bool = False,
        obstacle_randomization: PyBulletObstacleRandomizationSettings | None = None,
        reward_settings: PyBulletRewardSettings | None = None,
        curriculum: PyBulletCurriculumSettings | None = None,
        velocity_aviary_cls: Any | None = None,
        drone_model: Any | None = None,
        physics: Any | None = None,
    ) -> None:
        if obstacle_randomization is None:
            from swift.config import PyBulletObstacleRandomizationSettings

            obstacle_randomization = PyBulletObstacleRandomizationSettings()
        if reward_settings is None:
            from swift.config import PyBulletRewardSettings

            reward_settings = PyBulletRewardSettings()
        if curriculum is None:
            from swift.config import PyBulletCurriculumSettings

            curriculum = PyBulletCurriculumSettings()
        self.settings = settings or SimpleAvoidanceSettings()
        self._obstacle_randomization = obstacle_randomization
        self._reward_settings = reward_settings
        self._curriculum = curriculum
        self._obstacle_observation_slots = (
            int(obstacle_randomization.max_obstacles) if obstacle_randomization.enabled else 0
        )
        self.observation_shape = (15 + 4 * self._obstacle_observation_slots,)
        self._training_progress = 1.0
        self._active_curriculum_phase: PyBulletCurriculumPhaseSettings | None = None
        self._active_obstacles = self.settings.obstacles
        self._scenario_seed: int | None = None
        self._steps = 0
        self._path: list[Vector3] = []
        self._previous_goal_distance = 0.0
        self._initial_goal_distance = 1.0
        if runtime is None:
            from swift.sim.pybullet_runtime import PyBulletVelocityRuntimeEnv

            runtime = PyBulletVelocityRuntimeEnv(
                simulation_settings,
                max_speed=self.settings.max_speed,
                enable_obstacles=False,
                swift_obstacles=(
                    self.settings.obstacles
                    if enable_pybullet_obstacles and not obstacle_randomization.enabled
                    else ()
                ),
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
        self._active_curriculum_phase = self._snapshot_curriculum_phase()
        if self._obstacle_randomization.enabled:
            if seed is None:
                raise ValueError("seed is required when PyBullet obstacle randomization is enabled")
            self._scenario_seed = int(seed)
            active_randomization = self._obstacle_randomization
            phase = self._active_curriculum_phase
            if phase is not None and phase.max_obstacles == 0:
                self._active_obstacles = ()
            else:
                if phase is not None:
                    active_randomization = replace(
                        active_randomization,
                        min_obstacles=phase.min_obstacles,
                        max_obstacles=phase.max_obstacles,
                        require_path_blocker=phase.require_path_blocker,
                    )
                self._active_obstacles = sample_pybullet_obstacles(
                    active_randomization,
                    self.settings,
                    seed=self._scenario_seed,
                    phase_name=phase.name if phase is not None else "final",
                )
            setter = getattr(self._runtime, "set_swift_obstacles", None)
            if setter is None:
                raise RuntimeError("PyBullet runtime does not support randomized SWIFT obstacles")
            setter(self._active_obstacles)
        self._steps = 0
        observation, info = self._runtime.reset(seed=seed, options=options)
        observation = self._with_goal_tail(observation, info)
        self._path = [observation[0:3]]
        self._previous_goal_distance = float(observation[14])
        self._initial_goal_distance = max(self._previous_goal_distance, 1e-9)
        self._minimum_safety_distance = self._minimum_safety_distance_from(observation, info)
        return observation, self._info(
            info,
            reward_breakdown=RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
            reached_goal=False,
            collided=False,
            timed_out=False,
            minimum_safety_distance=self._minimum_safety_distance,
        )

    def set_training_progress(self, completed_timesteps: int, total_timesteps: int) -> None:
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if completed_timesteps < 0 or completed_timesteps > total_timesteps:
            raise ValueError("completed_timesteps must be between 0 and total_timesteps")
        self._training_progress = float(completed_timesteps) / float(total_timesteps)

    def step(
        self,
        action: DroneAction | Sequence[float],
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        drone_action = _coerce_action(action)
        observation, raw_reward, raw_terminated, raw_truncated, info = self._runtime.step(drone_action)
        observation = self._with_goal_tail(observation, info)
        self._steps += 1
        current_goal_distance = float(observation[14])
        current_safety_distance = self._minimum_safety_distance_from(observation, info)
        self._minimum_safety_distance = min(self._minimum_safety_distance, current_safety_distance)
        within_goal = current_goal_distance <= self.settings.goal_radius
        collided = bool(info.get("collided", False)) or _collided_from_safety(
            observation,
            current_safety_distance,
            self.settings.safety_margin,
        )
        reached_goal = bool(within_goal and not collided)
        terminated = bool(raw_terminated or reached_goal or collided)
        timed_out = (not terminated and not raw_truncated and self._steps >= self.settings.max_steps) or bool(
            info.get("timed_out", False)
        )
        truncated = bool(raw_truncated or timed_out)
        self._path.append(observation[0:3])
        normalized_approach = (self._previous_goal_distance - current_goal_distance) / self._initial_goal_distance
        reward_breakdown = RewardBreakdown(
            arrive=self._reward_settings.arrival_reward if reached_goal else 0.0,
            approach=normalized_approach * self._reward_settings.approach_scale,
            obstacle=-self._reward_settings.collision_penalty if collided else 0.0,
            smoothness=(
                -self._reward_settings.heading_smoothness_penalty
                * abs(float(drone_action.heading_delta))
            ),
            timeliness=(
                -self._reward_settings.episode_time_penalty / float(self.settings.max_steps)
                - (self._reward_settings.timeout_penalty if timed_out else 0.0)
            ),
        )
        if not math.isfinite(reward_breakdown.total):
            raise RuntimeError("PyBullet reward must be finite")
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
                collided=collided,
                timed_out=timed_out,
                minimum_safety_distance=self._minimum_safety_distance,
                raw_reward=float(raw_reward),
                episode_done=terminated or truncated,
            ),
        )

    def close(self) -> None:
        self._runtime.close()

    def _snapshot_curriculum_phase(self) -> PyBulletCurriculumPhaseSettings | None:
        if not self._curriculum.enabled:
            return None
        return self._curriculum.phase_for(self._training_progress)

    def _with_goal_tail(self, observation: tuple[float, ...], runtime_info: dict[str, Any]) -> tuple[float, ...]:
        position = observation[0:3]
        relative_goal = tuple(self.settings.goal[index] - position[index] for index in range(3))
        relative_obstacle, obstacle_radius = self._obstacle_tail(position, runtime_info)
        goal_distance = _distance(position, self.settings.goal)
        obstacle_slots = self._observation_obstacle_slots(position)
        return (
            *observation[0:7],
            *relative_goal,
            *relative_obstacle,
            obstacle_radius,
            goal_distance,
            *obstacle_slots,
        )

    def _obstacle_tail(
        self,
        position: Sequence[float],
        runtime_info: dict[str, Any],
    ) -> tuple[tuple[float, float, float], float]:
        runtime_relative = _vector3_from_info(runtime_info, "nearest_obstacle_relative")
        runtime_radius = _float_from_info(runtime_info, "nearest_obstacle_radius")
        if runtime_relative is not None and runtime_radius is not None:
            return runtime_relative, runtime_radius

        if not self._active_obstacles:
            return (0.0, 0.0, 0.0), 0.0

        nearest = min(
            self._active_obstacles,
            key=lambda obstacle: _distance(position, obstacle.position),
        )
        relative = tuple(float(nearest.position[index]) - float(position[index]) for index in range(3))
        return relative, float(nearest.radius)

    def _observation_obstacle_slots(self, position: Sequence[float]) -> tuple[float, ...]:
        if self._obstacle_observation_slots <= 0:
            return ()

        values: list[float] = []
        obstacles = sorted(
            self._active_obstacles,
            key=lambda obstacle: _distance(position, obstacle.position),
        )
        for obstacle in obstacles[: self._obstacle_observation_slots]:
            values.extend(
                (
                    float(obstacle.position[0]) - float(position[0]),
                    float(obstacle.position[1]) - float(position[1]),
                    float(obstacle.position[2]) - float(position[2]),
                    float(obstacle.radius),
                )
            )
        missing = self._obstacle_observation_slots - len(obstacles)
        values.extend((0.0, 0.0, 0.0, 0.0) * missing)
        return tuple(values)

    def _minimum_safety_distance_from(self, observation: tuple[float, ...], runtime_info: dict[str, Any]) -> float:
        runtime_value = _float_from_info(runtime_info, "minimum_safety_distance")
        if runtime_value is not None:
            return runtime_value
        obstacle_radius = float(observation[13])
        if obstacle_radius <= 0.0:
            return 0.0
        return _distance(observation[10:13], (0.0, 0.0, 0.0)) - obstacle_radius

    def _info(
        self,
        runtime_info: dict[str, Any],
        *,
        reward_breakdown: RewardBreakdown,
        reached_goal: bool,
        collided: bool,
        timed_out: bool,
        minimum_safety_distance: float,
        raw_reward: float | None = None,
        episode_done: bool = False,
    ) -> dict[str, Any]:
        info = dict(runtime_info)
        info.update(
            {
                "runtime_contract": "pybullet_velocity_training_compatibility",
                "steps": self._steps,
                "reached_goal": bool(reached_goal),
                "collided": bool(collided),
                "timed_out": bool(timed_out),
                "minimum_safety_distance": float(minimum_safety_distance),
                "reward_breakdown": reward_breakdown,
            }
        )
        if self._obstacle_randomization.enabled:
            info.update(
                {
                    "scenario_seed": self._scenario_seed,
                    "obstacle_count": len(self._active_obstacles),
                    "obstacles": self._active_obstacles,
                }
            )
        if self._active_curriculum_phase is not None:
            info.update(
                {
                    "curriculum_phase": self._active_curriculum_phase.name,
                    "curriculum_progress": self._training_progress,
                }
            )
        if raw_reward is not None:
            info["raw_reward"] = float(raw_reward)
        if episode_done:
            info["episode_metrics"] = EpisodeMetrics(
                reached_goal=bool(reached_goal),
                collided=bool(collided),
                timed_out=bool(timed_out),
                path_length=compute_path_length(tuple(self._path)),
                path_smoothness=compute_path_smoothness(tuple(self._path)),
                minimum_safety_distance=float(minimum_safety_distance),
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


def _vector3_from_info(info: dict[str, Any], key: str) -> tuple[float, float, float] | None:
    value = info.get(key)
    if value is None or isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _float_from_info(info: dict[str, Any], key: str) -> float | None:
    value = info.get(key)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _collided_from_safety(
    observation: tuple[float, ...],
    minimum_safety_distance: float,
    safety_margin: float,
) -> bool:
    obstacle_radius = float(observation[13])
    if obstacle_radius > 0.0:
        return minimum_safety_distance <= float(safety_margin)
    return minimum_safety_distance < 0.0
