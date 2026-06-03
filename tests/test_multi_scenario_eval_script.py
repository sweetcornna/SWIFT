import json
import subprocess
import sys
from pathlib import Path


def test_multi_scenario_eval_script_writes_report(tmp_path: Path):
    output = tmp_path / "multi_scenario.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multi_scenario_eval.py",
            "--candidate-limit",
            "2",
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
    assert report["record_type"] == "multi_scenario_evaluation_report"
    assert report["summary"]["scenario_count"] >= 2
    assert report["readiness"]["all_scenarios_evaluated"] is True
    assert f"SWIFT multi-scenario evaluation written: {output}" in result.stdout
