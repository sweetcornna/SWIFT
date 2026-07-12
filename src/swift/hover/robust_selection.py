"""Frozen held-out bank and lexicographic checkpoint-selection primitives."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

BANK_NAME = "heldout_robust_validation_v1"
BANK_CASE_COUNT = 100
BANK_HASH = "9412c6baf9a61d0085adfa56a882389158bd2d81683eab9e53e17bc9dc0725b0"
BOUNDS = np.array(((0.05, 0.25), (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10), (-3.0, 3.0), (-3.0, 3.0)), dtype=np.float64)
COEFFICIENTS = np.array(((17, 8), (19, 9), (23, 11), (29, 14), (31, 15), (37, 18)), dtype=np.int32)


def canonical_state_hash(values: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


def generate_validation_bank() -> np.ndarray:
    indices = np.arange(BANK_CASE_COUNT, dtype=np.int64)[:, None]
    ranks = (COEFFICIENTS[:, 0] * indices + COEFFICIENTS[:, 1]) % BANK_CASE_COUNT
    states = np.ascontiguousarray(BOUNDS[:, 0] + (BOUNDS[:, 1] - BOUNDS[:, 0]) * (ranks + 0.5) / 100.0, dtype=np.float64)
    if canonical_state_hash(states) != BANK_HASH:
        raise RuntimeError("frozen validation bank hash mismatch")
    return states


def nearest_rank_p95(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("p95 input must be a nonempty finite vector")
    return float(np.sort(array)[math.ceil(.95 * array.size) - 1])


def checkpoint_score(safety_completion_count: int, p95_mae: float, mean_return: float) -> tuple[int, float, float]:
    if isinstance(safety_completion_count, (bool, np.bool_)) or int(safety_completion_count) != safety_completion_count or safety_completion_count < 0:
        raise ValueError("safety completion count must be a nonnegative integer")
    if not math.isfinite(p95_mae) or not math.isfinite(mean_return):
        raise ValueError("checkpoint score fields must be finite")
    return int(safety_completion_count), -float(p95_mae), float(mean_return)


def select_checkpoint(safety_counts: Any, p95_maes: Any, mean_returns: Any) -> int:
    """Return the earliest row attaining a strict lexicographic improvement."""
    counts, maes, returns = map(np.asarray, (safety_counts, p95_maes, mean_returns))
    if counts.ndim != 1 or maes.shape != counts.shape or returns.shape != counts.shape or counts.size == 0:
        raise ValueError("checkpoint histories must be aligned nonempty vectors")
    best_score = None
    selected = -1
    for index in range(counts.size):
        score = checkpoint_score(int(counts[index]), float(maes[index]), float(returns[index]))
        if best_score is None or score > best_score:
            best_score, selected = score, index
    return selected
