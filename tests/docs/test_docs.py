# D:\project\SWIFT\tests\docs\test_docs.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_required_enterprise_docs_exist():
    required = [
        "docs/project-charter.md",
        "docs/architecture.md",
        "docs/implementation-roadmap.md",
        "docs/risk-register.md",
        "docs/source-materials.md",
    ]

    for relative in required:
        assert (ROOT / relative).exists(), f"Missing {relative}"


def test_project_charter_records_core_technical_line():
    text = (ROOT / "docs" / "project-charter.md").read_text(encoding="utf-8")

    assert "PPO" in text
    assert "HCA" in text
    assert "APF" in text
    assert "PyBullet" in text


def test_risk_register_covers_bootstrap_risks():
    text = (ROOT / "docs" / "risk-register.md").read_text(encoding="utf-8")

    for risk in ["Sparse reward", "PPO-HCA coupling", "APF local minima", "GPU availability"]:
        assert risk in text
