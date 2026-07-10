from swift.experiments.baseline_runner import BaselineRunConfig, run_baseline_episodes
from swift.experiments.artifacts import (
    ExperimentArtifactConfig,
    ExperimentArtifactPaths,
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
    checkpoint_filename,
    file_sha256,
)
from swift.experiments.ppo_checkpoint_evaluator import (
    PPOCheckpointEvaluationConfig,
    run_ppo_checkpoint_evaluation,
)
from swift.experiments.schema import ExperimentMetric, ExperimentSpec
from swift.experiments.stage1_report import build_stage1_report, write_stage1_report
from swift.experiments.tuning_runner import (
    PolicySearchSpace,
    Stage1Scenario,
    TuningRunConfig,
    evaluate_policy,
    run_stage1_policy_search,
)

__all__ = [
    "BaselineRunConfig",
    "ExperimentArtifactConfig",
    "ExperimentArtifactPaths",
    "ExperimentArtifactWriter",
    "ExperimentMetric",
    "ExperimentSpec",
    "PolicySearchSpace",
    "PPOCheckpointEvaluationConfig",
    "PyBulletCheckpointEvaluationConfig",
    "PyBulletGeometryValidationConfig",
    "PyBulletPPOTrainingRunConfig",
    "PyBulletMultiSeedTrainingConfig",
    "PyBulletMultiSeedCheckpointEvaluationConfig",
    "PyBulletRobustnessGateConfig",
    "PyBulletRobustnessThresholds",
    "Stage1Scenario",
    "TuningRunConfig",
    "artifact_reference",
    "build_artifact_manifest",
    "build_run_id",
    "build_stage1_report",
    "checkpoint_filename",
    "evaluate_policy",
    "file_sha256",
    "run_baseline_episodes",
    "run_ppo_checkpoint_evaluation",
    "run_pybullet_checkpoint_evaluation",
    "run_pybullet_geometry_validation",
    "run_pybullet_ppo_training",
    "run_pybullet_multi_seed_training",
    "run_pybullet_multi_seed_checkpoint_evaluation",
    "run_pybullet_robustness_gate",
    "run_stage1_policy_search",
    "write_stage1_report",
]


def __getattr__(name: str):
    if name in {"PyBulletGeometryValidationConfig", "run_pybullet_geometry_validation"}:
        from swift.experiments.pybullet_geometry_validator import (
            PyBulletGeometryValidationConfig,
            run_pybullet_geometry_validation,
        )

        exports = {
            "PyBulletGeometryValidationConfig": PyBulletGeometryValidationConfig,
            "run_pybullet_geometry_validation": run_pybullet_geometry_validation,
        }
        return exports[name]
    if name in {
        "PyBulletRobustnessGateConfig",
        "PyBulletRobustnessThresholds",
        "run_pybullet_robustness_gate",
    }:
        from swift.experiments.pybullet_robustness_gate import (
            PyBulletRobustnessGateConfig,
            PyBulletRobustnessThresholds,
            run_pybullet_robustness_gate,
        )

        exports = {
            "PyBulletRobustnessGateConfig": PyBulletRobustnessGateConfig,
            "PyBulletRobustnessThresholds": PyBulletRobustnessThresholds,
            "run_pybullet_robustness_gate": run_pybullet_robustness_gate,
        }
        return exports[name]
    if name in {
        "PyBulletMultiSeedCheckpointEvaluationConfig",
        "run_pybullet_multi_seed_checkpoint_evaluation",
    }:
        from swift.experiments.pybullet_multi_seed_checkpoint_evaluator import (
            PyBulletMultiSeedCheckpointEvaluationConfig,
            run_pybullet_multi_seed_checkpoint_evaluation,
        )

        exports = {
            "PyBulletMultiSeedCheckpointEvaluationConfig": PyBulletMultiSeedCheckpointEvaluationConfig,
            "run_pybullet_multi_seed_checkpoint_evaluation": run_pybullet_multi_seed_checkpoint_evaluation,
        }
        return exports[name]
    if name in {"PyBulletMultiSeedTrainingConfig", "run_pybullet_multi_seed_training"}:
        from swift.experiments.pybullet_multi_seed_training_runner import (
            PyBulletMultiSeedTrainingConfig,
            run_pybullet_multi_seed_training,
        )

        exports = {
            "PyBulletMultiSeedTrainingConfig": PyBulletMultiSeedTrainingConfig,
            "run_pybullet_multi_seed_training": run_pybullet_multi_seed_training,
        }
        return exports[name]
    if name in {"PyBulletCheckpointEvaluationConfig", "run_pybullet_checkpoint_evaluation"}:
        from swift.experiments.pybullet_checkpoint_evaluator import (
            PyBulletCheckpointEvaluationConfig,
            run_pybullet_checkpoint_evaluation,
        )

        exports = {
            "PyBulletCheckpointEvaluationConfig": PyBulletCheckpointEvaluationConfig,
            "run_pybullet_checkpoint_evaluation": run_pybullet_checkpoint_evaluation,
        }
        return exports[name]
    if name in {"PyBulletPPOTrainingRunConfig", "run_pybullet_ppo_training"}:
        from swift.experiments.pybullet_training_runner import (
            PyBulletPPOTrainingRunConfig,
            run_pybullet_ppo_training,
        )

        exports = {
            "PyBulletPPOTrainingRunConfig": PyBulletPPOTrainingRunConfig,
            "run_pybullet_ppo_training": run_pybullet_ppo_training,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
