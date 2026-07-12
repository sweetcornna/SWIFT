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


def test_readme_documents_pybullet_curriculum_seed_hygiene_and_gates():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in [
        "run_pybullet_geometry_check.py",
        "500000..500099",
        "1000000..1000099",
        "2000000..2000099",
        "pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k.json",
    ]:
        assert required in text

    assert "obsolete pre-accumulated-heading evidence" in text
    assert "Current randomized-final status: passed" in text
    assert "passed_gates=10/10" in text


def test_docs_define_robust_hover_boundary_and_curated_publication():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    combined = readme + architecture
    for required in [
        "ct_att_yawrate_v1",
        "heldout_robust_lexicographic_v1",
        "train_rms_physical12_v1",
        "artifacts/robust-hover/120k",
        "manifest.sha256.json",
        "does not replace or alter SWIFT's",
    ]:
        assert required in combined
    assert "not navigation or real-flight safety evidence" in combined
