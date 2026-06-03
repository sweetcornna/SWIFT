from pathlib import Path

import pytest

from swift.config import (
    ConvergenceGateProfile,
    EvaluationSettings,
    SimulationSettings,
    load_evaluation_settings,
    load_simulation_settings,
    load_yaml_file,
)


def test_load_yaml_file_returns_mapping(tmp_path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text("name: swift\ncount: 2\n", encoding="utf-8")

    assert load_yaml_file(config_path) == {"name": "swift", "count": 2}


def test_load_yaml_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_yaml_file(tmp_path / "missing.yaml")


def test_simulation_settings_resolve_default_paths(tmp_path):
    root = tmp_path / "pybullet"
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"pybullet_root: {root}",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks:",
                "  - test",
                "  - drone-demo",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_simulation_settings(config_path)

    assert settings == SimulationSettings(
        pybullet_root=root,
        pixi_executable=root / ".tools" / "pixi" / "pixi.exe",
        required_tasks=("test", "drone-demo"),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=120,
    )


def test_simulation_settings_require_tasks(tmp_path):
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pybullet_root: D:/project/pybullet",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks: []",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_tasks must contain at least one task"):
        load_simulation_settings(config_path)


def test_simulation_settings_rejects_scalar_required_tasks(tmp_path):
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pybullet_root: D:/project/pybullet",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks: test",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_tasks must be a list of task names"):
        load_simulation_settings(config_path)


def test_simulation_settings_rejects_non_string_task_names(tmp_path):
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pybullet_root: D:/project/pybullet",
                "pixi_executable: .tools/pixi/pixi.exe",
                "required_tasks:",
                "  - test",
                "  - 42",
                "check_task: test",
                "smoke_task: drone-demo",
                "command_timeout_seconds: 120",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_tasks must be a list of task names"):
        load_simulation_settings(config_path)


def test_load_evaluation_settings_parses_convergence_gate_profiles(tmp_path):
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "metrics:",
                "  - success_rate",
                "  - collision_rate",
                "artifacts:",
                "  root: outputs",
                "  episode_logs: outputs/episodes",
                "  experiment_reports: outputs/reports",
                "convergence_gate:",
                "  default_profile: long_training_convergence",
                "  profiles:",
                "    long_training_convergence:",
                "      required_evidence_level: long_training_convergence",
                "      required_variants: [ppo_mlp, ppo_hca, ppo_hca_apf]",
                "      min_success_rate: 0.95",
                "      max_collision_rate: 0.0",
                "      max_timeout_rate: 0.05",
                "      min_total_timesteps: 100000",
                "      min_episodes_completed: 100",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_evaluation_settings(config_path)

    assert settings == EvaluationSettings(
        metrics=("success_rate", "collision_rate"),
        artifact_root=Path("outputs"),
        episode_logs=Path("outputs/episodes"),
        experiment_reports=Path("outputs/reports"),
        default_convergence_profile="long_training_convergence",
        convergence_profiles={
            "long_training_convergence": ConvergenceGateProfile(
                required_evidence_level="long_training_convergence",
                required_variants=("ppo_mlp", "ppo_hca", "ppo_hca_apf"),
                min_success_rate=0.95,
                max_collision_rate=0.0,
                max_timeout_rate=0.05,
                min_total_timesteps=100000,
                min_episodes_completed=100,
            )
        },
    )


def test_current_evaluation_config_declares_enterprise_convergence_profile():
    settings = load_evaluation_settings(Path("configs") / "evaluation.yaml")

    assert set(settings.convergence_profiles) == {
        "cpu_smoke_ablation",
        "deterministic_multi_scenario_tuning",
        "long_training_convergence",
    }
    assert settings.convergence_profiles["cpu_smoke_ablation"].convergence_claim_allowed is False
    assert (
        settings.convergence_profiles["deterministic_multi_scenario_tuning"].convergence_claim_allowed
        is False
    )
    profile = settings.convergence_profiles["long_training_convergence"]
    assert settings.default_convergence_profile == "long_training_convergence"
    assert profile.convergence_claim_allowed is True
    assert profile.required_evidence_level == "long_training_convergence"
    assert profile.required_variants == ("ppo_mlp", "ppo_hca", "ppo_hca_apf")
    assert profile.min_success_rate == pytest.approx(0.95)
    assert profile.max_collision_rate == pytest.approx(0.0)
    assert profile.max_timeout_rate == pytest.approx(0.05)
    assert profile.min_total_timesteps >= 100000
    assert profile.min_episodes_completed >= 100


def test_evaluation_settings_rejects_unknown_default_convergence_profile(tmp_path):
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "convergence_gate:",
                "  default_profile: missing",
                "  profiles:",
                "    cpu_smoke_ablation:",
                "      required_evidence_level: cpu_smoke_ablation",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default convergence profile must exist"):
        load_evaluation_settings(config_path)


def test_non_long_convergence_profile_defaults_to_disallowing_convergence_claim(tmp_path):
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "convergence_gate:",
                "  default_profile: cpu_smoke_ablation",
                "  profiles:",
                "    cpu_smoke_ablation:",
                "      required_evidence_level: cpu_smoke_ablation",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_evaluation_settings(config_path)

    assert settings.convergence_profile("cpu_smoke_ablation").convergence_claim_allowed is False


def test_convergence_profile_parses_false_string_as_false(tmp_path):
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "convergence_gate:",
                "  default_profile: long_training_convergence",
                "  profiles:",
                "    long_training_convergence:",
                "      convergence_claim_allowed: 'false'",
                "      required_evidence_level: long_training_convergence",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_evaluation_settings(config_path)

    assert settings.convergence_profile("long_training_convergence").convergence_claim_allowed is False
