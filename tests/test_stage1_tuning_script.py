import json
import subprocess
import sys
from pathlib import Path


def test_stage1_tuning_script_writes_strict_json(tmp_path):
    output = tmp_path / "stage1_tuning.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage1_tuning.py",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    raw_metrics = output.read_text(encoding="utf-8")
    metrics = json.loads(raw_metrics)
    assert metrics["acceptance"]["accepted"] is True
    assert metrics["best_candidate"]["metrics"]["success"] is True
    assert "Infinity" not in raw_metrics
    assert "NaN" not in raw_metrics
    assert raw_metrics == json.dumps(metrics, allow_nan=False, indent=2, sort_keys=True) + "\n"
    assert f"SWIFT stage1 tuning written: {output}" in result.stdout
