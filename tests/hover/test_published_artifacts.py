import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "artifacts/robust-hover/120k"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_curated_publication_manifest_and_model_sidecars_are_exact():
    manifest = json.loads((PUBLICATION / "manifest.sha256.json").read_text(encoding="utf-8"))
    assert [item["seed"] for item in manifest["seeds"]] == [1, 2, 3]
    assert len(manifest["files"]) == 24
    assert all(not item["path"].endswith("evaluation.npz") for item in manifest["files"])
    for item in manifest["files"]:
        path = PUBLICATION / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]
    for seed in (1, 2, 3):
        directory = PUBLICATION / f"seed-{seed}"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        assert summary["training"]["train_seed"] == seed
        for kind in ("robust_best_model", "final_model"):
            binding = summary["model_normalization_bindings"][kind]
            assert sha256(directory / binding["model"]) == binding["model_sha256"]
            assert sha256(directory / binding["stats"]) == binding["stats_sha256"]
        for kind in ("robust_best_model", "final_model"):
            with np.load(directory / f"{kind}.obsnorm.npz", allow_pickle=False) as archive:
                assert archive.files
                assert all(archive[name].dtype.kind != "O" for name in archive.files)
        compact = json.loads((directory / "evaluation.compact.json").read_text(encoding="utf-8"))
        assert compact["verdict"] == "accepted"
        assert compact["evaluation_valid"] is True
        assert "cases" not in compact
