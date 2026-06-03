import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from swift.envs import SimpleAvoidanceEnv, SimpleAvoidanceSettings
from swift.rl.ppo import MLPActorCriticConfig, PPOTrainingConfig
from swift.rl.torch_ppo import train_ppo_mlp


def test_ppo_checkpoint_eval_script_writes_summary(tmp_path: Path):
    checkpoint_path = tmp_path / "ppo.ckpt"
    output_path = tmp_path / "eval.json"

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
            seed=5,
            network=MLPActorCriticConfig(hidden_sizes=(8,)),
            checkpoint_path=checkpoint_path,
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ppo_checkpoint_eval.py",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            "2",
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["record_type"] == "ppo_checkpoint_evaluation"
    assert summary["episodes_requested"] == 2
    assert f"SWIFT PPO checkpoint evaluation written: {output_path}" in result.stdout
