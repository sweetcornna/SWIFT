import json
from pathlib import Path

import numpy as np
import pytest

from swift.hover.contracts import HoverTrainingConfig, load_bound_training_summary, load_hover_training_config
from swift.hover.robust_selection import BANK_HASH, canonical_state_hash, generate_validation_bank, nearest_rank_p95, select_checkpoint


def test_repository_hover_config_loads_scientific_contract():
    root = Path(__file__).resolve().parents[2]
    config = load_hover_training_config(root / "configs/pybullet_hover.yaml")
    assert config == HoverTrainingConfig()


def test_scientific_100k_contract_rejects_sparse_validation():
    with pytest.raises(ValueError, match="scientific"):
        HoverTrainingConfig(eval_frequency=20_000)


def test_curated_summary_default_validates_models_that_are_present():
    root = Path(__file__).resolve().parents[2]
    summary = load_bound_training_summary(root / "artifacts/robust-hover/120k/seed-1")
    assert set(summary["model_normalization_bindings"]).issuperset({"robust_best_model", "final_model"})


def test_frozen_validation_bank_and_lexicographic_selection():
    bank = generate_validation_bank()
    assert bank.shape == (100, 6)
    assert canonical_state_hash(bank) == BANK_HASH
    assert nearest_rank_p95(np.arange(1.0, 101.0)) == 95.0
    assert select_checkpoint([99, 100, 100, 100], [0.01, 0.08, 0.07, 0.07], [500, 100, 90, 100]) == 3
