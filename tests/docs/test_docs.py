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


def test_docs_define_evidence_profile_boundaries():
    combined = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in [
            "docs/architecture.md",
            "docs/implementation-roadmap.md",
            "docs/risk-register.md",
        ]
    )

    for profile in [
        "cpu_smoke_ablation",
        "deterministic_multi_scenario_tuning",
        "long_training_convergence",
    ]:
        assert profile in combined

    assert "cpu_smoke_ablation is not convergence evidence" in combined
    assert "deterministic_multi_scenario_tuning is not convergence evidence" in combined
    assert "only long_training_convergence may support a convergence claim" in combined


def test_readme_documents_pybullet_ppo_training_entrypoint():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "scripts\\bootstrap.ps1" in text
    assert "scripts/bootstrap.py" in text
    assert "run_pybullet_ppo_training.py" in text
    assert r"D:\project\.venvs\swift-pybullet-pixi" in text
    assert "pybullet_velocity_training_compatibility" in text
