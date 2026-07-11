from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swift.config import TrainingSettings


TASK_CONTRACT_SCHEMA_VERSION = 1


def build_pybullet_task_contract(settings: TrainingSettings) -> dict[str, Any]:
    contract_settings = {
        "environment": asdict(settings.environment),
        "policy": asdict(settings.policy),
        "obstacle_randomization": asdict(settings.pybullet_obstacle_randomization),
        "reward": asdict(settings.pybullet_reward),
        "curriculum": asdict(settings.pybullet_curriculum),
    }
    canonical = json.dumps(
        contract_settings,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    normalized_settings = json.loads(canonical)
    return {
        "schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "hash": hash_pybullet_task_contract_settings(normalized_settings),
        "settings": normalized_settings,
    }


def hash_pybullet_task_contract_settings(settings: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(settings),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
