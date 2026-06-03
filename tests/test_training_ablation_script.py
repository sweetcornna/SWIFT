import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_training_ablation_script_writes_ranked_report(tmp_path: Path):
    output = tmp_path / "ablation.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_training_ablation.py",
            "--total-timesteps",
            "32",
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
    assert report["record_type"] == "training_ablation_report"
    assert report["readiness"]["all_variants_completed"] is True
    assert report["best_variant"] in {"ppo_mlp", "ppo_hca", "ppo_hca_apf"}
    assert f"SWIFT training ablation written: {output}" in result.stdout
