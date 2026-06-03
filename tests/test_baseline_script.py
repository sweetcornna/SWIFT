import json
import subprocess
import sys
from pathlib import Path


def test_baseline_demo_writes_metrics(tmp_path):
    output = tmp_path / "baseline_metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_baseline_demo.py",
            "--episodes",
            "3",
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
    assert metrics["episodes"] == 3
    assert "success_rate" in metrics
    assert "Infinity" not in raw_metrics
    assert "NaN" not in raw_metrics
    assert f"SWIFT baseline metrics written: {output}" in result.stdout
