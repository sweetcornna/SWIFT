import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings
from swift.experiments.ppo_checkpoint_evaluator import (
    PPOCheckpointEvaluationConfig,
    run_ppo_checkpoint_evaluation,
)
from swift.rl.ppo import MLPActorCriticConfig, PPOTrainingConfig
from swift.rl.torch_ppo import train_ppo_mlp


def test_ppo_checkpoint_evaluation_writes_strict_summary(tmp_path: Path):
    checkpoint_path = tmp_path / "ppo.ckpt"
    output_path = tmp_path / "checkpoint_eval.json"

    def make_env():
        return SimpleAvoidanceEnv(SimpleAvoidanceSettings(max_steps=8, goal=(4.0, 0.0, 0.0)))

    train_ppo_mlp(
        make_env,
        PPOTrainingConfig(
            total_timesteps=64,
            rollout_steps=32,
            minibatch_size=16,
            update_epochs=1,
            torch_num_threads=1,
            seed=9,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
            checkpoint_path=checkpoint_path,
        ),
    )

    summary = run_ppo_checkpoint_evaluation(
        PPOCheckpointEvaluationConfig(
            checkpoint_path=checkpoint_path,
            output=output_path,
            episodes=2,
            seed=21,
            environment=SimpleAvoidanceSettings(max_steps=8, goal=(4.0, 0.0, 0.0)),
        )
    )

    raw_summary = output_path.read_text(encoding="utf-8")
    assert json.loads(raw_summary) == summary
    assert summary["schema_version"] == 1
    assert summary["record_type"] == "ppo_checkpoint_evaluation"
    assert summary["checkpoint_path"] == str(checkpoint_path)
    assert summary["episodes_requested"] == 2
    assert len(summary["episodes"]) == 2
    assert summary["artifacts"]["summary_json"] == str(output_path)
    assert "NaN" not in raw_summary
    assert "Infinity" not in raw_summary
    for value in summary["metrics"].values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_ppo_checkpoint_evaluation_rejects_non_positive_episode_count(tmp_path: Path):
    with pytest.raises(ValueError, match="episodes"):
        PPOCheckpointEvaluationConfig(checkpoint_path=tmp_path / "missing.ckpt", episodes=0)
