from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import math

from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
)
from swift.experiments.tuning_runner import (
    PolicySearchSpace,
    Stage1Scenario,
    TuningRunConfig,
    run_stage1_policy_search,
)
from swift.envs import SimpleAvoidanceEnv
from swift.rl.mlp_baseline import MLPBaselinePolicy, MLPBaselinePolicyConfig


DEFAULT_BOUNDARY_NOTES = (
    "headless_metrics_are_source_of_truth",
    "videos_are_illustrative_when_present",
    "deterministic_tuning_does_not_claim_training_convergence",
    "simulator_evidence_is_not_real_flight_safety_evidence",
)


@dataclass(frozen=True)
class ScenarioCase:
    identifier: str
    scenario: Stage1Scenario
    seed: int = 0
    minimum_safety_distance: float | None = None
    building_density: str | None = None
    dynamic_drone_count: int = 0
    delivery_target_set: str | None = None
    altitude_profile: str | None = None

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        if not identifier:
            raise ValueError("scenario identifier must not be blank")
        if self.seed < 0:
            raise ValueError("scenario seed must be non-negative")
        if self.minimum_safety_distance is not None:
            minimum_safety_distance = float(self.minimum_safety_distance)
            if not math.isfinite(minimum_safety_distance) or minimum_safety_distance < 0.0:
                raise ValueError("scenario minimum_safety_distance must be finite and non-negative")
            object.__setattr__(self, "minimum_safety_distance", minimum_safety_distance)
        if self.dynamic_drone_count < 0:
            raise ValueError("dynamic_drone_count must be non-negative")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "dynamic_drone_count", int(self.dynamic_drone_count))


@dataclass(frozen=True)
class MultiScenarioEvaluationConfig:
    scenarios: tuple[ScenarioCase, ...] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    search_space: PolicySearchSpace = field(default_factory=PolicySearchSpace)
    minimum_safety_distance: float = 0.30
    candidate_limit: int | None = None
    output: Path | None = None
    stage: str = "stage4"
    variant: str = "multi_scenario_policy_tuning"

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ValueError("scenarios must contain at least one scenario")
        identifiers = [scenario.identifier for scenario in scenarios]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("scenario identifiers must be unique")
        minimum_safety_distance = float(self.minimum_safety_distance)
        if not math.isfinite(minimum_safety_distance) or minimum_safety_distance < 0.0:
            raise ValueError("minimum_safety_distance must be finite and non-negative")
        if self.candidate_limit is not None and self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive or None")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "minimum_safety_distance", minimum_safety_distance)
        object.__setattr__(self, "stage", str(self.stage).strip() or "stage4")
        object.__setattr__(self, "variant", str(self.variant).strip() or "multi_scenario_policy_tuning")


DEFAULT_SCENARIOS = (
    ScenarioCase(
        identifier="fixed-obstacle",
        scenario=Stage1Scenario(),
        building_density="single_obstacle",
        delivery_target_set="single_goal",
        altitude_profile="level_delivery",
    ),
    ScenarioCase(
        identifier="open-corridor",
        scenario=Stage1Scenario(obstacles=(), max_steps=30),
        building_density="open",
        delivery_target_set="single_goal",
        altitude_profile="level_delivery",
    ),
    ScenarioCase(
        identifier="offset-obstacle",
        scenario=Stage1Scenario(
            goal=(8.0, 2.0, 0.0),
            obstacles=(),
            max_steps=70,
        ),
        building_density="open_offset",
        delivery_target_set="single_offset_goal",
        altitude_profile="level_delivery",
    ),
)


def run_multi_scenario_evaluation(config: MultiScenarioEvaluationConfig | None = None) -> dict[str, Any]:
    run_config = config or MultiScenarioEvaluationConfig()
    writer = ExperimentArtifactWriter()
    config_hash = _config_hash(run_config)
    run_id = build_run_id(
        stage=run_config.stage,
        variant=run_config.variant,
        seed=_run_seed(run_config.scenarios),
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(run_config.stage, run_config.variant, run_id)
    output_path = run_config.output or paths.summary_json
    manifest_path = output_path.with_suffix(".manifest.json") if run_config.output is not None else paths.manifest_json
    metrics_table_path = output_path.with_suffix(".metrics.json")
    trajectory_path = (
        output_path.with_suffix(".trajectories.jsonl") if run_config.output is not None else paths.episode_jsonl
    )
    scenario_results = [_evaluate_scenario(case, run_config) for case in run_config.scenarios]
    scenario_reports = [result["report"] for result in scenario_results]
    ranking = _global_candidate_ranking(scenario_results)
    summary = _summary(scenario_reports, ranking)
    lineage = {
        "config_hash": config_hash,
        "evaluation_level": "deterministic_multi_scenario_tuning",
        "evidence_level": "deterministic_multi_scenario_tuning",
        "convergence_claim": False,
        "policy_search": "stage1_deterministic_mlp_policy_search",
        "source_runner": "swift.experiments.tuning_runner.run_stage1_policy_search",
        "simulator": "swift.simple_avoidance",
        "trajectory_capture": "deterministic_rerun_of_selected_policy",
        "seed_semantics": "simple_avoidance_seed_recorded_for_provenance_only",
    }
    report = {
        "schema_version": 1,
        "record_type": "multi_scenario_evaluation_report",
        "report_scope": "stage4_multi_scenario_evaluation",
        "evaluation_level": "deterministic_multi_scenario_tuning",
        "stage": run_config.stage,
        "variant": run_config.variant,
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lineage": lineage,
        "search_space": _search_space_dict(run_config.search_space),
        "ranking_formula": {
            "sort": (
                "accepted_all_desc_success_rate_desc_collision_rate_asc_timeout_rate_asc_"
                "average_path_length_asc_average_path_smoothness_asc_minimum_safety_distance_desc_config_asc"
            ),
            "convergence_claim": False,
        },
        "summary": summary,
        "ranking": ranking,
        "scenarios": scenario_reports,
        "readiness": {
            "all_scenarios_evaluated": len(scenario_reports) == len(run_config.scenarios),
            "all_scenarios_have_accepted_candidate": summary["accepted_scenario_count"] == summary["scenario_count"],
            "global_candidate_accepted": summary["globally_accepted_candidate_count"] > 0,
            "convergence_claim": False,
            "evidence_level": "deterministic_multi_scenario_tuning",
        },
        "interpretation": {
            "apf_contribution_claim": "not_claimed",
            "ablation_required_for_apf_claim": True,
        },
        "boundary_notes": list(DEFAULT_BOUNDARY_NOTES),
        "artifact_refs": {
            "summary_json": str(output_path),
            "metrics_table_json": str(metrics_table_path),
            "trajectory_jsonl": str(trajectory_path),
            "manifest_json": str(manifest_path),
        },
        "artifacts": {
            "summary_json": str(output_path),
            "metrics_table_json": str(metrics_table_path),
            "trajectory_jsonl": str(trajectory_path),
            "manifest_json": str(manifest_path),
        },
    }
    metrics_table = _metrics_table(run_id, run_config, scenario_reports)
    trajectory_records = _trajectory_records(run_id, run_config, scenario_results)
    writer.write_summary(metrics_table_path, metrics_table)
    _write_jsonl(trajectory_path, trajectory_records)
    writer.write_summary(output_path, report)
    writer.write_manifest(
        manifest_path,
        build_artifact_manifest(
            subject_record_type=report["record_type"],
            run_id=run_id,
            stage=run_config.stage,
            variant=run_config.variant,
            lineage=lineage,
            outputs=[
                artifact_reference(output_path, role="multi_scenario_report"),
                artifact_reference(metrics_table_path, role="metrics_table_json"),
                artifact_reference(trajectory_path, role="trajectory_jsonl"),
            ],
        ),
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _evaluate_scenario(case: ScenarioCase, config: MultiScenarioEvaluationConfig) -> dict[str, Any]:
    minimum_safety_distance = _effective_safety_distance(case, config.minimum_safety_distance)
    result = run_stage1_policy_search(
        TuningRunConfig(
            scenario=case.scenario,
            search_space=config.search_space,
            minimum_safety_distance=minimum_safety_distance,
            candidate_limit=None,
        )
    )
    best_candidate = result["best_candidate"]
    metrics = _scenario_metrics(best_candidate, result["baseline"])
    full_candidates = list(result["candidates"])
    if config.candidate_limit is not None:
        result["candidates"] = full_candidates[: config.candidate_limit]
    report = {
        "scenario_id": case.identifier,
        "scenario": result["scenario"],
        "scenario_matrix": _scenario_matrix(case),
        "run_provenance": {
            "seed": case.seed,
            "episode_count": 1,
            "max_steps": case.scenario.max_steps,
            "environment_adapter": "swift.envs.SimpleAvoidanceEnv",
            "policy_family": "deterministic_mlp_policy_search",
            "candidate_count": result["acceptance"]["candidate_count"],
            "minimum_safety_distance": minimum_safety_distance,
            "trajectory_capture": "deterministic_rerun_of_selected_policy",
            "seed_semantics": "simple_avoidance_seed_recorded_for_provenance_only",
        },
        "baseline": result["baseline"],
        "best_candidate": best_candidate,
        "candidates": result["candidates"],
        "acceptance": result["acceptance"],
        "metrics": metrics,
        "failure_breakdown": _failure_breakdown(metrics, minimum_safety_distance),
    }
    return {
        "case": case,
        "scenario_id": case.identifier,
        "report": report,
        "full_candidates": full_candidates,
    }


def _metrics_table(
    run_id: str,
    config: MultiScenarioEvaluationConfig,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "multi_scenario_metrics_table",
        "stage": config.stage,
        "variant": config.variant,
        "run_id": run_id,
        "evaluation_level": "deterministic_multi_scenario_tuning",
        "rows": [
            {
                "scenario_id": scenario["scenario_id"],
                "scenario_matrix": scenario["scenario_matrix"],
                "run_provenance": scenario["run_provenance"],
                "acceptance": scenario["acceptance"],
                "metrics": scenario["metrics"],
                "failure_breakdown": scenario["failure_breakdown"],
            }
            for scenario in scenarios
        ],
    }


def _trajectory_records(
    run_id: str,
    config: MultiScenarioEvaluationConfig,
    scenario_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario_result in scenario_results:
        case = scenario_result["case"]
        report = scenario_result["report"]
        assert isinstance(case, ScenarioCase)
        assert isinstance(report, dict)
        policy_config = _trajectory_policy_config(report)
        env_settings = case.scenario.build_settings().replace(
            max_speed=policy_config.max_speed,
            max_climb_rate=policy_config.max_climb_rate,
        )
        env = SimpleAvoidanceEnv(env_settings)
        policy = MLPBaselinePolicy(policy_config)
        observation, _ = env.reset(seed=case.seed)
        terminated = False
        truncated = False
        step_index = 0
        while not terminated and not truncated:
            action = policy.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            drone_state = info["drone_state"]
            reward_breakdown = info["reward_breakdown"]
            record = {
                "schema_version": 1,
                "record_type": "scenario_trajectory_step",
                "stage": config.stage,
                "variant": config.variant,
                "run_id": run_id,
                "scenario_id": case.identifier,
                "step_index": step_index,
                "state": {
                    "position": list(drone_state.position),
                    "velocity": list(drone_state.velocity),
                    "yaw": float(drone_state.yaw),
                },
                "action": asdict(action),
                "reward": float(reward),
                "reward_breakdown": asdict(reward_breakdown),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "goal_distance": float(observation[14]),
                "obstacles": [
                    {
                        "position": list(obstacle.position),
                        "radius": float(obstacle.radius),
                        "velocity": list(obstacle.velocity),
                    }
                    for obstacle in info["obstacles"]
                ],
            }
            if "episode_metrics" in info:
                record["episode_metrics"] = asdict(info["episode_metrics"])
            records.append(record)
            step_index += 1
    return records


def _trajectory_policy_config(report: dict[str, Any]) -> MLPBaselinePolicyConfig:
    best_candidate = report["best_candidate"]
    if isinstance(best_candidate, dict):
        config = best_candidate["config"]
    else:
        config = report["baseline"]["config"]
    assert isinstance(config, dict)
    return MLPBaselinePolicyConfig(**config)


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, allow_nan=False, sort_keys=True) + "\n" for record in records)
    target.write_text(payload, encoding="utf-8")


def _scenario_matrix(case: ScenarioCase) -> dict[str, Any]:
    return {
        "building_density": case.building_density or _building_density(case.scenario),
        "dynamic_drone_count": case.dynamic_drone_count,
        "delivery_target_set": case.delivery_target_set or _delivery_target_set(case.scenario),
        "altitude_profile": case.altitude_profile or _altitude_profile(case.scenario),
    }


def _scenario_metrics(best_candidate: object, baseline: object) -> dict[str, Any]:
    if isinstance(best_candidate, dict):
        metrics = best_candidate["metrics"]
    else:
        metrics = baseline
    assert isinstance(metrics, dict)
    return {
        "success": bool(metrics["success"]),
        "collided": bool(metrics["collided"]),
        "timed_out": bool(metrics["timed_out"]),
        "path_length": float(metrics["path_length"]),
        "path_smoothness": float(metrics["path_smoothness"]),
        "minimum_safety_distance": float(metrics["minimum_safety_distance"]),
        "steps": int(metrics["steps"]),
    }


def _global_candidate_ranking(scenario_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, dict[str, Any]] = {}
    for scenario_result in scenario_results:
        scenario_id = str(scenario_result["scenario_id"])
        for candidate in scenario_result["full_candidates"]:
            assert isinstance(candidate, dict)
            config = candidate["config"]
            assert isinstance(config, dict)
            key = _config_key(config)
            record = by_config.setdefault(
                key,
                {
                    "config": dict(config),
                    "scenarios": [],
                },
            )
            metrics = candidate["metrics"]
            assert isinstance(metrics, dict)
            record["scenarios"].append(
                {
                    "scenario_id": scenario_id,
                    "accepted": bool(candidate["accepted"]),
                    "metrics": _candidate_metrics(metrics),
                }
            )

    ranked = sorted(by_config.values(), key=_global_candidate_sort_key)
    output = []
    for rank, record in enumerate(ranked, start=1):
        scenario_metrics = [scenario["metrics"] for scenario in record["scenarios"]]
        metrics = {
            "success_rate": _rate(scenario_metrics, "success"),
            "collision_rate": _rate(scenario_metrics, "collided"),
            "timeout_rate": _rate(scenario_metrics, "timed_out"),
            "average_path_length": _average(scenario_metrics, "path_length"),
            "average_path_smoothness": _average(scenario_metrics, "path_smoothness"),
            "minimum_safety_distance": min(float(metric["minimum_safety_distance"]) for metric in scenario_metrics),
        }
        output.append(
            {
                "rank": rank,
                "config": record["config"],
                "accepted_all": all(bool(scenario["accepted"]) for scenario in record["scenarios"]),
                "metrics": metrics,
                "scenarios": record["scenarios"],
            }
        )
    return output


def _summary(scenarios: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> dict[str, Any]:
    globally_accepted = [candidate for candidate in ranking if candidate["accepted_all"]]
    return {
        "scenario_count": len(scenarios),
        "candidate_count": len(ranking),
        "globally_accepted_candidate_count": len(globally_accepted),
        "best_config": ranking[0]["config"] if ranking else None,
        "accepted_scenario_count": sum(1 for scenario in scenarios if scenario["acceptance"]["accepted"]),
        "best_success_count": sum(1 for scenario in scenarios if scenario["metrics"]["success"]),
        "baseline_collision_count": sum(1 for scenario in scenarios if scenario["baseline"]["collided"]),
        "timeout_count": sum(1 for scenario in scenarios if scenario["metrics"]["timed_out"]),
    }


def _global_candidate_sort_key(candidate: dict[str, Any]) -> tuple[object, ...]:
    scenario_metrics = [scenario["metrics"] for scenario in candidate["scenarios"]]
    metrics = {
        "success_rate": _rate(scenario_metrics, "success"),
        "collision_rate": _rate(scenario_metrics, "collided"),
        "timeout_rate": _rate(scenario_metrics, "timed_out"),
        "average_path_length": _average(scenario_metrics, "path_length"),
        "average_path_smoothness": _average(scenario_metrics, "path_smoothness"),
        "minimum_safety_distance": min(float(metric["minimum_safety_distance"]) for metric in scenario_metrics),
    }
    return (
        0 if all(bool(scenario["accepted"]) for scenario in candidate["scenarios"]) else 1,
        -metrics["success_rate"],
        metrics["collision_rate"],
        metrics["timeout_rate"],
        metrics["average_path_length"],
        metrics["average_path_smoothness"],
        -metrics["minimum_safety_distance"],
        _config_key(candidate["config"]),
    )


def _candidate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(metrics["success"]),
        "collided": bool(metrics["collided"]),
        "timed_out": bool(metrics["timed_out"]),
        "steps": int(metrics["steps"]),
        "path_length": float(metrics["path_length"]),
        "path_smoothness": float(metrics["path_smoothness"]),
        "minimum_safety_distance": float(metrics["minimum_safety_distance"]),
    }


def _rate(metrics: list[dict[str, Any]], name: str) -> float:
    return sum(1 for metric in metrics if bool(metric[name])) / len(metrics)


def _average(metrics: list[dict[str, Any]], name: str) -> float:
    return sum(float(metric[name]) for metric in metrics) / len(metrics)


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, allow_nan=False, sort_keys=True)


def _effective_safety_distance(case: ScenarioCase, default: float) -> float:
    if case.minimum_safety_distance is not None:
        return float(case.minimum_safety_distance)
    if not case.scenario.obstacles:
        return 0.0
    return default


def _failure_breakdown(metrics: dict[str, Any], minimum_safety_distance: float) -> dict[str, int]:
    return {
        "collisions": int(bool(metrics["collided"])),
        "timeouts": int(bool(metrics["timed_out"])),
        "unsafe_distance": int(float(metrics["minimum_safety_distance"]) < minimum_safety_distance),
        "stuck_or_local_minimum": int(bool(metrics["timed_out"]) and not bool(metrics["collided"])),
        "altitude_preference_violations": 0,
    }


def _building_density(scenario: Stage1Scenario) -> str:
    obstacle_count = len(scenario.obstacles)
    if obstacle_count == 0:
        return "open"
    if obstacle_count == 1:
        return "single_obstacle"
    if obstacle_count <= 3:
        return "sparse"
    return "dense"


def _delivery_target_set(scenario: Stage1Scenario) -> str:
    return "single_goal" if scenario.goal[1] == 0.0 else "single_offset_goal"


def _altitude_profile(scenario: Stage1Scenario) -> str:
    if scenario.start[2] == scenario.goal[2]:
        return "level_delivery"
    return "altitude_change"


def _run_seed(scenarios: tuple[ScenarioCase, ...]) -> int:
    return min(case.seed for case in scenarios)


def _search_space_dict(search_space: PolicySearchSpace) -> dict[str, list[float]]:
    return {
        "max_speed": list(search_space.max_speed),
        "max_heading_delta": list(search_space.max_heading_delta),
        "obstacle_avoidance_distance": list(search_space.obstacle_avoidance_distance),
        "avoidance_heading_delta": list(search_space.avoidance_heading_delta),
        "max_climb_rate": list(search_space.max_climb_rate),
    }


def _config_hash(config: MultiScenarioEvaluationConfig) -> str:
    material = repr(
        (
            config.scenarios,
            config.search_space,
            config.minimum_safety_distance,
            config.candidate_limit,
            config.stage,
            config.variant,
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
