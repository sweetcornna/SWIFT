import json
import subprocess
import sys
from pathlib import Path


def test_convergence_gate_script_writes_report(tmp_path: Path):
    input_path = tmp_path / "ablation.json"
    output_path = tmp_path / "gate.json"
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
