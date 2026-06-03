import json
import subprocess
import sys
from pathlib import Path


def test_convergence_gate_script_writes_report(tmp_path: Path):
    input_path = tmp_path / "holdout.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(json.dumps(_holdout_report()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_convergence_gate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--required-evidence-level",
            "long_training_convergence",
            "--min-seed-count",
            "1",
            "--min-total-timesteps",
            "4096",
            "--min-episodes-completed",
            "16",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["record_type"] == "training_convergence_gate_report"
    assert report["readiness"]["convergence_claim"] is True
    assert f"SWIFT convergence gate written: {output_path}" in result.stdout
    assert "convergence_claim=True" in result.stdout


def test_convergence_gate_script_defaults_to_enterprise_evaluation_profile(tmp_path: Path):
    input_path = tmp_path / "single_seed_short.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(
        json.dumps(
            {
                "record_type": "training_ablation_report",
                "run_id": "stage4_ablation_seed-0_cfg-test_20260603t000000z",
                "lineage": {"evidence_level": "long_training_convergence"},
                "best_variant": "ppo_hca",
                "seed": 0,
                "variants": [
                    _variant("ppo_mlp"),
                    _variant("ppo_hca"),
                    _variant("ppo_hca_apf"),
                ],
                "readiness": {"evidence_level": "long_training_convergence"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_convergence_gate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["readiness"]["convergence_claim"] is False
    assert failed["seed_count"]["expected"] == ">=3"
    assert failed["total_timesteps"]["expected"] == ">=100000"
    assert "convergence_claim=False" in result.stdout


def test_convergence_gate_script_uses_configured_profile_thresholds(tmp_path: Path):
    input_path = tmp_path / "ablation.json"
    output_path = tmp_path / "gate.json"
    config_path = tmp_path / "evaluation.yaml"
    input_path.write_text(
        json.dumps(
            {
                "record_type": "training_ablation_report",
                "run_id": "stage4_ablation_seed-0_cfg-test_20260603t000000z",
                "lineage": {"evidence_level": "long_training_convergence"},
                "best_variant": "ppo_hca",
                "variants": [
                    _variant("ppo_mlp"),
                    _variant("ppo_hca"),
                    _variant("ppo_hca_apf"),
                ],
                "readiness": {"evidence_level": "long_training_convergence"},
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_convergence_gate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--profile",
            "long_training_convergence",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["readiness"]["convergence_claim"] is False
    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert failed["total_timesteps"]["expected"] == ">=100000"
    assert failed["episodes_completed"]["expected"] == ">=100"
    assert "convergence_claim=False" in result.stdout


def test_convergence_gate_script_audits_cpu_smoke_profile_without_convergence_claim(tmp_path: Path):
    input_path = tmp_path / "smoke_ablation.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(
        json.dumps(
            {
                "record_type": "training_ablation_report",
                "run_id": "stage4_ablation_seed-0_cfg-test_20260603t000000z",
                "lineage": {"evidence_level": "cpu_smoke_ablation"},
                "best_variant": "ppo_hca",
                "variants": [
                    _variant("ppo_mlp"),
                    _variant("ppo_hca"),
                    _variant("ppo_hca_apf"),
                ],
                "readiness": {"evidence_level": "cpu_smoke_ablation"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_convergence_gate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            "configs/evaluation.yaml",
            "--profile",
            "cpu_smoke_ablation",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["readiness"]["convergence_claim"] is False
    assert failed["convergence_claim_allowed"]["actual"] is False
    assert "convergence_claim=False" in result.stdout


def test_convergence_gate_script_audits_multi_scenario_profile_without_schema_error(tmp_path: Path):
    input_path = tmp_path / "multi_scenario.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(
        json.dumps(
            {
                "record_type": "multi_scenario_evaluation_report",
                "run_id": "stage4_multi-scenario-policy-tuning_seed-0_cfg-test_20260603t000000z",
                "stage": "stage4",
                "variant": "multi_scenario_policy_tuning",
                "lineage": {"evidence_level": "deterministic_multi_scenario_tuning"},
                "summary": {"scenario_count": 2, "globally_accepted_candidate_count": 1},
                "ranking": [
                    {
                        "rank": 1,
                        "accepted_all": True,
                        "metrics": {
                            "success_rate": 1.0,
                            "collision_rate": 0.0,
                            "timeout_rate": 0.0,
                            "average_path_length": 8.0,
                            "average_path_smoothness": 0.2,
                            "minimum_safety_distance": 0.4,
                        },
                    }
                ],
                "scenarios": [
                    {"scenario_id": "fixed", "metrics": {"steps": 12}},
                    {"scenario_id": "open", "metrics": {"steps": 8}},
                ],
                "readiness": {"evidence_level": "deterministic_multi_scenario_tuning"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_convergence_gate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--profile",
            "deterministic_multi_scenario_tuning",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["source_record_type"] == "multi_scenario_evaluation_report"
    assert report["readiness"]["convergence_claim"] is False
    assert failed["convergence_claim_allowed"]["actual"] is False
    assert "convergence_claim=False" in result.stdout


def test_convergence_gate_script_can_exit_nonzero_when_gate_rejects(tmp_path: Path):
    input_path = tmp_path / "smoke_ablation.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(
        json.dumps(
            {
                "record_type": "training_ablation_report",
                "run_id": "stage4_ablation_seed-0_cfg-test_20260603t000000z",
                "lineage": {"evidence_level": "cpu_smoke_ablation"},
                "best_variant": "ppo_hca",
                "variants": [
                    _variant("ppo_mlp"),
                    _variant("ppo_hca"),
                    _variant("ppo_hca_apf"),
                ],
                "readiness": {"evidence_level": "cpu_smoke_ablation"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_convergence_gate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            "configs/evaluation.yaml",
            "--profile",
            "cpu_smoke_ablation",
            "--fail-on-reject",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "convergence_claim=False" in result.stdout
    assert output_path.exists()


def test_convergence_gate_script_non_long_override_cannot_claim_convergence(tmp_path: Path):
    input_path = tmp_path / "smoke_ablation.json"
    output_path = tmp_path / "gate.json"
    input_path.write_text(
        json.dumps(
            {
                "record_type": "training_ablation_report",
                "run_id": "stage4_ablation_seed-0_cfg-test_20260603t000000z",
                "lineage": {"evidence_level": "cpu_smoke_ablation"},
                "best_variant": "ppo_hca",
                "variants": [
                    _variant("ppo_mlp"),
                    _variant("ppo_hca"),
                    _variant("ppo_hca_apf"),
                ],
                "readiness": {"evidence_level": "cpu_smoke_ablation"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_convergence_gate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--required-evidence-level",
            "cpu_smoke_ablation",
            "--min-success-rate",
            "0.0",
            "--max-collision-rate",
            "1.0",
            "--max-timeout-rate",
            "1.0",
            "--min-total-timesteps",
            "64",
            "--min-episodes-completed",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    failed = {gate["name"]: gate for gate in report["gates"] if not gate["passed"]}
    assert report["readiness"]["convergence_claim"] is False
    assert failed["convergence_claim_allowed"]["actual"] is False
    assert "convergence_claim=False" in result.stdout


def _variant(name: str) -> dict[str, object]:
    return {
        "variant": name,
        "completed": True,
        "metrics": {
            "success_rate": 1.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "average_episode_return": 12.0,
        },
        "training": {
            "total_timesteps": 4096,
            "updates": 8,
            "episodes_completed": 16,
        },
    }


def _holdout_report() -> dict[str, object]:
    metrics = {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "average_episode_return": 12.0,
    }
    return {
        "record_type": "multi_seed_checkpoint_holdout_report",
        "run_id": "stage4_multi-seed-checkpoint-holdout_seed-10000_cfg-test_20260603t000000z",
        "lineage": {"evidence_level": "long_training_convergence"},
        "best_variant": "ppo_hca",
        "seed_count": 1,
        "training_seeds": [0],
        "holdout_seeds": [10000],
        "variants": [
            _holdout_variant("ppo_mlp", metrics),
            _holdout_variant("ppo_hca", metrics),
            _holdout_variant("ppo_hca_apf", metrics),
        ],
        "readiness": {"evidence_level": "long_training_convergence"},
    }


def _holdout_variant(name: str, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "variant": name,
        "completed": True,
        "metrics": metrics,
        "training": {
            "total_timesteps": 4096,
            "updates": 8,
            "episodes_completed": 16,
        },
        "seed_metrics": [
            {
                "training_seed": 0,
                "holdout_seed": 10000,
                "metrics": metrics,
                "training": {
                    "total_timesteps": 4096,
                    "updates": 8,
                    "episodes_completed": 16,
                },
            }
        ],
    }
