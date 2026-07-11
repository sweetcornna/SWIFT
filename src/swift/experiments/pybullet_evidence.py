from __future__ import annotations


DEVELOPMENT_SEED_START = 500_000
DEVELOPMENT_SEED_END = 500_099
CONSUMED_SEED_START = 1_000_000
CONSUMED_SEED_END = 1_000_099
FINAL_HOLDOUT_SEED_START = 2_000_000
FINAL_HOLDOUT_SEED_END = 2_000_099


def classify_pybullet_holdout(seed_start: int, seed_end: int) -> str:
    if seed_start == FINAL_HOLDOUT_SEED_START and seed_end == FINAL_HOLDOUT_SEED_END:
        return "reserved_final"
    if DEVELOPMENT_SEED_START <= seed_start <= seed_end <= DEVELOPMENT_SEED_END:
        return "development"
    if CONSUMED_SEED_START <= seed_start <= seed_end <= CONSUMED_SEED_END:
        return "consumed"
    if FINAL_HOLDOUT_SEED_START <= seed_start <= seed_end <= FINAL_HOLDOUT_SEED_END:
        return "reserved_final_partial"
    return "unclassified"
