from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import yaml

from swift.core import ObstacleState
from swift.envs import SimpleAvoidanceSettings
from swift.experiments.artifacts import ExperimentArtifactConfig
from swift.rl import APFConfig, MLPBaselinePolicyConfig, PPOConfig


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return data


@dataclass(frozen=True)
class SimulationSettings:
    pybullet_root: Path
    pixi_executable: Path
    required_tasks: tuple[str, ...]
    check_task: str
    smoke_task: str
    command_timeout_seconds: int

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SimulationSettings":
        pybullet_root = Path(str(mapping["pybullet_root"])).expanduser()
        raw_pixi = Path(str(mapping["pixi_executable"]))
        pixi_executable = raw_pixi if raw_pixi.is_absolute() else pybullet_root / raw_pixi

        raw_required_tasks = mapping.get("required_tasks", ())
        if isinstance(raw_required_tasks, str) or not isinstance(
            raw_required_tasks, Sequence
        ):
            raise ValueError("required_tasks must be a list of task names")
        if any(not isinstance(task, str) or not task.strip() for task in raw_required_tasks):
            raise ValueError("required_tasks must be a list of task names")
        required_tasks = tuple(task for task in raw_required_tasks)
        if not required_tasks:
            raise ValueError("required_tasks must contain at least one task")

        timeout = int(mapping.get("command_timeout_seconds", 120))
        if timeout <= 0:
            raise ValueError("command_timeout_seconds must be positive")

        return cls(
            pybullet_root=pybullet_root,
            pixi_executable=pixi_executable,
            required_tasks=required_tasks,
            check_task=str(mapping.get("check_task", "test")),
            smoke_task=str(mapping.get("smoke_task", "drone-demo")),
            command_timeout_seconds=timeout,
        )


def load_simulation_settings(path: str | Path) -> SimulationSettings:
    return SimulationSettings.from_mapping(load_yaml_file(path))


@dataclass(frozen=True)
class TrainingRunSettings:
    stage: str = "stage1"
    variant: str = "ppo_mlp"
    seed: int = 0
    episodes: int = 10
    total_timesteps: int = 1024

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainingRunSettings":
        settings = cls(
            stage=str(mapping.get("stage", cls.stage)).strip() or cls.stage,
            variant=str(mapping.get("variant", cls.variant)).strip() or cls.variant,
            seed=int(mapping.get("seed", cls.seed)),
            episodes=_positive_int("episodes", mapping.get("episodes", cls.episodes)),
            total_timesteps=_positive_int(
                "total_timesteps",
                mapping.get("total_timesteps", cls.total_timesteps),
            ),
        )
        return settings


@dataclass(frozen=True)
class BaselineMetadata:
    model: str = "MLP"
    purpose: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BaselineMetadata":
        return cls(
            model=str(mapping.get("model", cls.model)).strip() or cls.model,
            purpose=str(mapping.get("purpose", cls.purpose)),
        )


@dataclass(frozen=True)
class PyBulletObstacleRandomizationSettings:
    enabled: bool = False
    min_obstacles: int = 1
    max_obstacles: int = 3
    x_range: tuple[float, float] = (0.12, 0.38)
    y_range: tuple[float, float] = (-0.30, 0.30)
    z: float = 0.1125
    radius_range: tuple[float, float] = (0.04, 0.08)
    endpoint_clearance: float = 0.02
    inter_obstacle_clearance: float = 0.02
    vehicle_radius: float = 0.0
    require_path_blocker: bool = True
    max_sampling_attempts: int = 256

    def __post_init__(self) -> None:
        min_obstacles = _positive_int("min_obstacles", self.min_obstacles)
        max_obstacles = _positive_int("max_obstacles", self.max_obstacles)
        if max_obstacles < min_obstacles:
            raise ValueError("max_obstacles must be >= min_obstacles")
        x_range = _finite_range("x_range", self.x_range)
        y_range = _finite_range("y_range", self.y_range)
        radius_range = _finite_range("radius_range", self.radius_range, positive=True)
        z = float(self.z)
        if not math.isfinite(z):
            raise ValueError("z must be finite")
        object.__setattr__(self, "enabled", _bool_value("enabled", self.enabled))
        object.__setattr__(self, "min_obstacles", min_obstacles)
        object.__setattr__(self, "max_obstacles", max_obstacles)
        object.__setattr__(self, "x_range", x_range)
        object.__setattr__(self, "y_range", y_range)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "radius_range", radius_range)
        object.__setattr__(
            self,
            "endpoint_clearance",
            _non_negative_float("endpoint_clearance", self.endpoint_clearance),
        )
        object.__setattr__(
            self,
            "inter_obstacle_clearance",
            _non_negative_float("inter_obstacle_clearance", self.inter_obstacle_clearance),
        )
        object.__setattr__(self, "vehicle_radius", _non_negative_float("vehicle_radius", self.vehicle_radius))
        object.__setattr__(
            self,
            "require_path_blocker",
            _bool_value("require_path_blocker", self.require_path_blocker),
        )
        object.__setattr__(
            self,
            "max_sampling_attempts",
            _positive_int("max_sampling_attempts", self.max_sampling_attempts),
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PyBulletObstacleRandomizationSettings":
        defaults = cls()
        return cls(
            enabled=mapping.get("enabled", defaults.enabled),
            min_obstacles=mapping.get("min_obstacles", defaults.min_obstacles),
            max_obstacles=mapping.get("max_obstacles", defaults.max_obstacles),
            x_range=mapping.get("x_range", defaults.x_range),
            y_range=mapping.get("y_range", defaults.y_range),
            z=mapping.get("z", defaults.z),
            radius_range=mapping.get("radius_range", defaults.radius_range),
            endpoint_clearance=mapping.get("endpoint_clearance", defaults.endpoint_clearance),
            inter_obstacle_clearance=mapping.get(
                "inter_obstacle_clearance",
                defaults.inter_obstacle_clearance,
            ),
            vehicle_radius=mapping.get("vehicle_radius", defaults.vehicle_radius),
            require_path_blocker=mapping.get("require_path_blocker", defaults.require_path_blocker),
            max_sampling_attempts=mapping.get("max_sampling_attempts", defaults.max_sampling_attempts),
        )


@dataclass(frozen=True)
class PyBulletRewardSettings:
    arrival_reward: float = 100.0
    approach_scale: float = 1.0
    collision_penalty: float = 100.0
    timeout_penalty: float = 0.0
    episode_time_penalty: float = 1.0
    heading_smoothness_penalty: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "arrival_reward",
            "approach_scale",
            "collision_penalty",
            "timeout_penalty",
            "episode_time_penalty",
            "heading_smoothness_penalty",
        ):
            object.__setattr__(self, name, _non_negative_float(name, getattr(self, name)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PyBulletRewardSettings":
        defaults = cls()
        return cls(
            arrival_reward=mapping.get("arrival_reward", defaults.arrival_reward),
            approach_scale=mapping.get("approach_scale", defaults.approach_scale),
            collision_penalty=mapping.get("collision_penalty", defaults.collision_penalty),
            timeout_penalty=mapping.get("timeout_penalty", defaults.timeout_penalty),
            episode_time_penalty=mapping.get("episode_time_penalty", defaults.episode_time_penalty),
            heading_smoothness_penalty=mapping.get(
                "heading_smoothness_penalty",
                defaults.heading_smoothness_penalty,
            ),
        )


@dataclass(frozen=True)
class PyBulletAPFActionPriorSettings:
    enabled: bool = False
    attractive_gain: float = 1.0
    repulsive_gain: float = 1.0
    influence_radius: float = 2.0
    max_repulsive_magnitude: float = 10.0
    epsilon: float = 1e-6
    ignore_obstacles_behind: bool = False
    bypass_enabled: bool = False
    bypass_lateral_offset: float = 0.25
    bypass_forward_margin: float = 0.08
    bypass_clearance: float = 0.16
    visibility_planner_enabled: bool = False
    visibility_clearance: float = 0.18
    visibility_samples: int = 16
    policy_residual_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _bool_value("enabled", self.enabled))
        object.__setattr__(self, "attractive_gain", _non_negative_float("attractive_gain", self.attractive_gain))
        object.__setattr__(self, "repulsive_gain", _non_negative_float("repulsive_gain", self.repulsive_gain))
        object.__setattr__(self, "influence_radius", _positive_float("influence_radius", self.influence_radius))
        object.__setattr__(
            self,
            "max_repulsive_magnitude",
            _positive_float("max_repulsive_magnitude", self.max_repulsive_magnitude),
        )
        object.__setattr__(self, "epsilon", _positive_float("epsilon", self.epsilon))
        object.__setattr__(
            self,
            "ignore_obstacles_behind",
            _bool_value("ignore_obstacles_behind", self.ignore_obstacles_behind),
        )
        object.__setattr__(self, "bypass_enabled", _bool_value("bypass_enabled", self.bypass_enabled))
        object.__setattr__(
            self,
            "bypass_lateral_offset",
            _positive_float("bypass_lateral_offset", self.bypass_lateral_offset),
        )
        object.__setattr__(
            self,
            "bypass_forward_margin",
            _non_negative_float("bypass_forward_margin", self.bypass_forward_margin),
        )
        object.__setattr__(
            self,
            "bypass_clearance",
            _non_negative_float("bypass_clearance", self.bypass_clearance),
        )
        object.__setattr__(
            self,
            "visibility_planner_enabled",
            _bool_value("visibility_planner_enabled", self.visibility_planner_enabled),
        )
        object.__setattr__(
            self,
            "visibility_clearance",
            _positive_float("visibility_clearance", self.visibility_clearance),
        )
        visibility_samples = _positive_int("visibility_samples", self.visibility_samples)
        if visibility_samples < 8:
            raise ValueError("visibility_samples must be at least 8")
        object.__setattr__(self, "visibility_samples", visibility_samples)
        object.__setattr__(
            self,
            "policy_residual_scale",
            _unit_interval_float("policy_residual_scale", self.policy_residual_scale),
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PyBulletAPFActionPriorSettings":
        defaults = cls()
        return cls(
            enabled=mapping.get("enabled", defaults.enabled),
            attractive_gain=mapping.get("attractive_gain", defaults.attractive_gain),
            repulsive_gain=mapping.get("repulsive_gain", defaults.repulsive_gain),
            influence_radius=mapping.get("influence_radius", defaults.influence_radius),
            max_repulsive_magnitude=mapping.get(
                "max_repulsive_magnitude",
                defaults.max_repulsive_magnitude,
            ),
            epsilon=mapping.get("epsilon", defaults.epsilon),
            ignore_obstacles_behind=mapping.get(
                "ignore_obstacles_behind",
                defaults.ignore_obstacles_behind,
            ),
            bypass_enabled=mapping.get("bypass_enabled", defaults.bypass_enabled),
            bypass_lateral_offset=mapping.get(
                "bypass_lateral_offset",
                defaults.bypass_lateral_offset,
            ),
            bypass_forward_margin=mapping.get(
                "bypass_forward_margin",
                defaults.bypass_forward_margin,
            ),
            bypass_clearance=mapping.get("bypass_clearance", defaults.bypass_clearance),
            visibility_planner_enabled=mapping.get(
                "visibility_planner_enabled",
                defaults.visibility_planner_enabled,
            ),
            visibility_clearance=mapping.get(
                "visibility_clearance",
                defaults.visibility_clearance,
            ),
            visibility_samples=mapping.get("visibility_samples", defaults.visibility_samples),
            policy_residual_scale=mapping.get(
                "policy_residual_scale",
                defaults.policy_residual_scale,
            ),
        )

    def to_apf_config(self) -> APFConfig | None:
        if not self.enabled:
            return None
        return APFConfig(
            attractive_gain=self.attractive_gain,
            repulsive_gain=self.repulsive_gain,
            influence_radius=self.influence_radius,
            max_repulsive_magnitude=self.max_repulsive_magnitude,
            epsilon=self.epsilon,
            ignore_obstacles_behind=self.ignore_obstacles_behind,
            bypass_enabled=self.bypass_enabled,
            bypass_lateral_offset=self.bypass_lateral_offset,
            bypass_forward_margin=self.bypass_forward_margin,
            bypass_clearance=self.bypass_clearance,
            visibility_planner_enabled=self.visibility_planner_enabled,
            visibility_clearance=self.visibility_clearance,
            visibility_samples=self.visibility_samples,
            policy_residual_scale=self.policy_residual_scale,
        )


@dataclass(frozen=True)
class PyBulletCurriculumPhaseSettings:
    name: str
    end_fraction: float
    min_obstacles: int
    max_obstacles: int
    require_path_blocker: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("curriculum phase name must not be empty")
        end_fraction = _unit_interval("end_fraction", self.end_fraction)
        min_obstacles = _non_negative_int("min_obstacles", self.min_obstacles)
        max_obstacles = _non_negative_int("max_obstacles", self.max_obstacles)
        if max_obstacles < min_obstacles:
            raise ValueError("max_obstacles must be >= min_obstacles")
        require_path_blocker = _bool_value("require_path_blocker", self.require_path_blocker)
        if require_path_blocker and max_obstacles == 0:
            raise ValueError("path blocker requires at least one obstacle")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "end_fraction", end_fraction)
        object.__setattr__(self, "min_obstacles", min_obstacles)
        object.__setattr__(self, "max_obstacles", max_obstacles)
        object.__setattr__(self, "require_path_blocker", require_path_blocker)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PyBulletCurriculumPhaseSettings":
        return cls(
            name=mapping.get("name", ""),
            end_fraction=mapping.get("end_fraction", 0.0),
            min_obstacles=mapping.get("min_obstacles", 0),
            max_obstacles=mapping.get("max_obstacles", 0),
            require_path_blocker=mapping.get("require_path_blocker", False),
        )


@dataclass(frozen=True)
class PyBulletCurriculumSettings:
    enabled: bool = False
    phases: tuple[PyBulletCurriculumPhaseSettings, ...] = ()

    def __post_init__(self) -> None:
        enabled = _bool_value("enabled", self.enabled)
        phases = tuple(self.phases)
        if enabled and not phases:
            raise ValueError("phases must not be empty")
        boundaries = tuple(phase.end_fraction for phase in phases)
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("end_fraction values must be strictly increasing")
        if phases and not math.isclose(boundaries[-1], 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("final curriculum end_fraction must equal 1.0")
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "phases", phases)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PyBulletCurriculumSettings":
        raw_phases = mapping.get("phases", ())
        if raw_phases is None:
            raw_phases = ()
        if isinstance(raw_phases, str) or not isinstance(raw_phases, Sequence):
            raise ValueError("phases must be a list")
        phases = []
        for phase in raw_phases:
            if not isinstance(phase, Mapping):
                raise ValueError("curriculum phases must be mappings")
            phases.append(PyBulletCurriculumPhaseSettings.from_mapping(phase))
        return cls(enabled=mapping.get("enabled", False), phases=tuple(phases))

    def phase_for(self, progress: float) -> PyBulletCurriculumPhaseSettings:
        numeric = _unit_interval_float("progress", progress)
        if not self.enabled or not self.phases:
            raise ValueError("curriculum is not enabled")
        return next((phase for phase in self.phases if numeric < phase.end_fraction), self.phases[-1])


@dataclass(frozen=True)
class TrainingSettings:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    policy: MLPBaselinePolicyConfig = field(default_factory=MLPBaselinePolicyConfig)
    environment: SimpleAvoidanceSettings = field(default_factory=SimpleAvoidanceSettings)
    run: TrainingRunSettings = field(default_factory=TrainingRunSettings)
    artifact: ExperimentArtifactConfig = field(default_factory=ExperimentArtifactConfig)
    baseline: BaselineMetadata = field(default_factory=BaselineMetadata)
    pybullet_obstacle_randomization: PyBulletObstacleRandomizationSettings = field(
        default_factory=PyBulletObstacleRandomizationSettings
    )
    pybullet_reward: PyBulletRewardSettings = field(default_factory=PyBulletRewardSettings)
    pybullet_apf_action_prior: PyBulletAPFActionPriorSettings = field(
        default_factory=PyBulletAPFActionPriorSettings
    )
    pybullet_curriculum: PyBulletCurriculumSettings = field(default_factory=PyBulletCurriculumSettings)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainingSettings":
        return cls(
            ppo=_ppo_from_mapping(_mapping_section(mapping, "ppo")),
            policy=_policy_from_mapping(_mapping_section(mapping, "policy")),
            environment=_environment_from_mapping(_mapping_section(mapping, "environment")),
            run=TrainingRunSettings.from_mapping(_mapping_section(mapping, "run")),
            artifact=_artifact_from_mapping(_artifact_section(mapping)),
            baseline=BaselineMetadata.from_mapping(_mapping_section(mapping, "baseline")),
            pybullet_obstacle_randomization=PyBulletObstacleRandomizationSettings.from_mapping(
                _mapping_section(mapping, "pybullet_obstacle_randomization")
            ),
            pybullet_reward=PyBulletRewardSettings.from_mapping(_mapping_section(mapping, "pybullet_reward")),
            pybullet_apf_action_prior=PyBulletAPFActionPriorSettings.from_mapping(
                _mapping_section(mapping, "pybullet_apf_action_prior")
            ),
            pybullet_curriculum=PyBulletCurriculumSettings.from_mapping(
                _mapping_section(mapping, "pybullet_curriculum")
            ),
        )


def load_training_settings(path: str | Path) -> TrainingSettings:
    return TrainingSettings.from_mapping(load_yaml_file(path))


@dataclass(frozen=True)
class ConvergenceGateProfile:
    convergence_claim_allowed: bool = True
    required_evidence_level: str = "long_training_convergence"
    required_variants: tuple[str, ...] = ("ppo_mlp", "ppo_hca", "ppo_hca_apf")
    min_success_rate: float = 0.95
    max_collision_rate: float = 0.0
    max_timeout_rate: float = 0.05
    min_seed_count: int = 1
    min_total_timesteps: int = 4096
    min_episodes_completed: int = 10

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ConvergenceGateProfile":
        defaults = cls()
        required_evidence_level = str(
            mapping.get("required_evidence_level", defaults.required_evidence_level)
        ).strip() or defaults.required_evidence_level
        default_claim_allowed = required_evidence_level == "long_training_convergence"
        required_variants = mapping.get("required_variants", defaults.required_variants)
        if isinstance(required_variants, str) or not isinstance(required_variants, Sequence):
            raise ValueError("required_variants must be a list of variant names")
        variants = tuple(str(variant).strip() for variant in required_variants if str(variant).strip())
        if not variants:
            raise ValueError("required_variants must contain at least one variant")
        return cls(
            convergence_claim_allowed=_bool_value(
                "convergence_claim_allowed",
                mapping.get("convergence_claim_allowed", default_claim_allowed),
            ),
            required_evidence_level=required_evidence_level,
            required_variants=variants,
            min_success_rate=_unit_interval_float(
                "min_success_rate",
                mapping.get("min_success_rate", defaults.min_success_rate),
            ),
            max_collision_rate=_unit_interval_float(
                "max_collision_rate",
                mapping.get("max_collision_rate", defaults.max_collision_rate),
            ),
            max_timeout_rate=_unit_interval_float(
                "max_timeout_rate",
                mapping.get("max_timeout_rate", defaults.max_timeout_rate),
            ),
            min_seed_count=_positive_int(
                "min_seed_count",
                mapping.get("min_seed_count", defaults.min_seed_count),
            ),
            min_total_timesteps=_positive_int(
                "min_total_timesteps",
                mapping.get("min_total_timesteps", defaults.min_total_timesteps),
            ),
            min_episodes_completed=_positive_int(
                "min_episodes_completed",
                mapping.get("min_episodes_completed", defaults.min_episodes_completed),
            ),
        )


@dataclass(frozen=True)
class EvaluationSettings:
    metrics: tuple[str, ...] = ()
    artifact_root: Path = Path("outputs")
    episode_logs: Path = Path("outputs/episodes")
    experiment_reports: Path = Path("outputs/reports")
    default_convergence_profile: str = "long_training_convergence"
    convergence_profiles: dict[str, ConvergenceGateProfile] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EvaluationSettings":
        metrics = _string_tuple("metrics", mapping.get("metrics", ()))
        artifacts = _mapping_section(mapping, "artifacts")
        gate = _mapping_section(mapping, "convergence_gate")
        profiles_mapping = _mapping_section(gate, "profiles")
        profiles = {
            str(name).strip(): ConvergenceGateProfile.from_mapping(_profile_mapping(profile))
            for name, profile in profiles_mapping.items()
            if str(name).strip()
        }
        default_profile = str(gate.get("default_profile", cls.default_convergence_profile)).strip()
        if profiles and default_profile not in profiles:
            raise ValueError("default convergence profile must exist in convergence_gate.profiles")
        return cls(
            metrics=metrics,
            artifact_root=Path(str(artifacts.get("root", cls.artifact_root))).expanduser(),
            episode_logs=Path(str(artifacts.get("episode_logs", cls.episode_logs))).expanduser(),
            experiment_reports=Path(str(artifacts.get("experiment_reports", cls.experiment_reports))).expanduser(),
            default_convergence_profile=default_profile,
            convergence_profiles=profiles,
        )

    def convergence_profile(self, name: str | None = None) -> ConvergenceGateProfile:
        profile_name = str(name or self.default_convergence_profile).strip()
        try:
            return self.convergence_profiles[profile_name]
        except KeyError as exc:
            raise ValueError(f"unknown convergence profile: {profile_name}") from exc


def load_evaluation_settings(path: str | Path) -> EvaluationSettings:
    return EvaluationSettings.from_mapping(load_yaml_file(path))


def _mapping_section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _profile_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("convergence profile must be a mapping")
    return value


def _string_tuple(name: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of strings")
    values = tuple(str(item).strip() for item in value if str(item).strip())
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return values


def _artifact_section(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    if "artifact" in mapping:
        return _mapping_section(mapping, "artifact")
    return _mapping_section(mapping, "artifacts")


def _ppo_from_mapping(mapping: Mapping[str, Any]) -> PPOConfig:
    defaults = PPOConfig()
    return PPOConfig(
        rollout_steps=_positive_int("rollout_steps", mapping.get("rollout_steps", defaults.rollout_steps)),
        minibatch_size=_positive_int("minibatch_size", mapping.get("minibatch_size", defaults.minibatch_size)),
        update_epochs=_positive_int("update_epochs", mapping.get("update_epochs", defaults.update_epochs)),
        clip_range=_positive_float("clip_range", mapping.get("clip_range", defaults.clip_range)),
        gamma=_unit_interval("gamma", mapping.get("gamma", defaults.gamma)),
        gae_lambda=_unit_interval("gae_lambda", mapping.get("gae_lambda", defaults.gae_lambda)),
        entropy_coef=_non_negative_float("entropy_coef", mapping.get("entropy_coef", defaults.entropy_coef)),
        value_loss_coef=_non_negative_float("value_loss_coef", mapping.get("value_loss_coef", defaults.value_loss_coef)),
        max_grad_norm=_positive_float("max_grad_norm", mapping.get("max_grad_norm", defaults.max_grad_norm)),
    )


def _policy_from_mapping(mapping: Mapping[str, Any]) -> MLPBaselinePolicyConfig:
    defaults = MLPBaselinePolicyConfig()
    return MLPBaselinePolicyConfig(
        max_speed=float(mapping.get("max_speed", defaults.max_speed)),
        max_heading_delta=float(mapping.get("max_heading_delta", defaults.max_heading_delta)),
        min_speed_fraction=float(mapping.get("min_speed_fraction", defaults.min_speed_fraction)),
        max_climb_rate=float(mapping.get("max_climb_rate", defaults.max_climb_rate)),
        obstacle_avoidance_distance=float(
            mapping.get("obstacle_avoidance_distance", defaults.obstacle_avoidance_distance)
        ),
        avoidance_heading_delta=float(mapping.get("avoidance_heading_delta", defaults.avoidance_heading_delta)),
    )


def _environment_from_mapping(mapping: Mapping[str, Any]) -> SimpleAvoidanceSettings:
    values = dict(mapping)
    if "obstacles" in values:
        values["obstacles"] = _obstacles_from_sequence(values["obstacles"])
    return SimpleAvoidanceSettings(**values)


def _obstacles_from_sequence(value: Any) -> tuple[ObstacleState, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("environment.obstacles must be a list of mappings")

    obstacles: list[ObstacleState] = []
    for obstacle in value:
        if not isinstance(obstacle, Mapping):
            raise ValueError("environment.obstacles must be a list of mappings")
        if "position" not in obstacle:
            raise ValueError("environment obstacle position is required")
        if "radius" not in obstacle:
            raise ValueError("environment obstacle radius is required")
        obstacles.append(
            ObstacleState(
                position=tuple(obstacle["position"]),
                radius=float(obstacle["radius"]),
                velocity=tuple(obstacle.get("velocity", (0.0, 0.0, 0.0))),
            )
        )
    return tuple(obstacles)


def _artifact_from_mapping(mapping: Mapping[str, Any]) -> ExperimentArtifactConfig:
    defaults = ExperimentArtifactConfig()
    return ExperimentArtifactConfig(
        root=Path(str(mapping.get("root", defaults.root))).expanduser(),
        episode_logs=Path(str(mapping.get("episode_logs", defaults.episode_logs))).expanduser(),
        experiment_reports=Path(str(mapping.get("experiment_reports", defaults.experiment_reports))).expanduser(),
        checkpoints=Path(str(mapping.get("checkpoints", defaults.checkpoints))).expanduser(),
    )


def _positive_int(name: str, value: Any) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _non_negative_int(name: str, value: Any) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _positive_float(name: str, value: Any) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _non_negative_float(name: str, value: Any) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _unit_interval(name: str, value: Any) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be greater than 0 and <= 1")
    return numeric


def _unit_interval_float(name: str, value: Any) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return numeric


def _bool_value(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _finite_range(
    name: str,
    value: Sequence[float],
    *,
    positive: bool = False,
) -> tuple[float, float]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 values")
    lower, upper = float(value[0]), float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} values must be finite")
    if positive and (lower <= 0.0 or upper <= 0.0):
        raise ValueError(f"{name} values must be positive")
    if lower > upper:
        raise ValueError(f"{name} lower value must be <= upper value")
    return lower, upper
