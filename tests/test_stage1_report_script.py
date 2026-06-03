import json
import subprocess
import sys
from pathlib import Path


def test_stage1_report_script_writes_report(tmp_path: Path):
    ppo_summary = tmp_path / "ppo.json"
    tuning_summary = tmp_path / "tuning.json"
    output = tmp_path / "stage1_report.json"
    ppo_summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "ppo_training_smoke",
                "stage": "stage1",
                "variant": "ppo_mlp",
                "run_id": "stage1_ppo_seed-7",
                "training": {"total_timesteps": 64, "updates": 1, "episodes_completed": 4},
                "metrics": {
                    "success_rate": 0.0,
                    "collision_rate": 0.75,
                    "timeout_rate": 0.25,
                    "average_episode_return": -4.0,
                },
                "artifacts": {
                    "summary_json": str(ppo_summary),
                    "training_history_jsonl": "history.jsonl",
                    "checkpoint_path": "model.ckpt",
                },
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    tuning_summary.write_text(
        json.dumps(
            {
                "baseline": {"success": False, "collided": True, "timed_out": False},
                "best_candidate": {
                    "accepted": True,
                    "metrics": {"success": True, "collided": False, "timed_out": False},
                },
                "acceptance": {"accepted": True, "candidate_count": 4, "accepted_candidate_count": 1},
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_report.py",
            "--ppo-summary",
            str(ppo_summary),
            "--tuning-summary",
            str(tuning_summary),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["record_type"] == "stage1_training_tuning_report"
    assert report["readiness"]["collision_to_success"] is True
    assert f"SWIFT Stage 1 report written: {output}" in result.stdout
