from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json

from swift.config import TrainingSettings
from swift.envs import SimpleAvoidanceEnv
from swift.experiments.artifacts import ExperimentArtifactWriter, build_run_id, checkpoint_filename
from swift.rl.hca import HCAActorCriticConfig
from swift.rl.ppo import HCAPPOTrainingConfig, PPOTrainingResult, train_ppo_hca


@dataclass(frozen=True)
class HCATrainingRunConfig:
    settings: TrainingSettings
    total_timesteps: int | None = None
    seed: int | None = None
    output: Path | None = None
    stage: str = "stage2"
    variant: str = "ppo_hca"

    def __post_init__(self) -> None:
        if self.total_timesteps is not None and self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "stage", str(self.stage).strip() or "stage2")
        object.__setattr__(self, "variant", str(self.variant).strip() or "ppo_hca")


def run_hca_training_smoke(config: HCATrainingRunConfig) -> dict[str, Any]:
    base_training_config = _training_config(config)
    writer = ExperimentArtifactWriter(config.settings.artifact)
    run_id = build_run_id(
        stage=config.stage,
        variant=config.variant,
        seed=base_training_config.seed,
        config_hash=_config_hash(config.settings, base_training_config.total_timesteps, config.stage, config.variant),
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(config.stage, config.variant, run_id)
    output_path = config.output or paths.summary_json
    checkpoint_path = paths.checkpoint_dir / checkpoint_filename(
        variant=config.variant,
        run_id=run_id,
        episode=0,
        step=base_training_config.total_timesteps,
        label="final",
        metric_name="planned_steps",
        metric_value=float(base_training_config.total_timesteps),
    )
    training_config = _training_config(
        config,
        checkpoint_path=checkpoint_path,
        history_path=paths.episode_jsonl,
    )

    result = train_ppo_hca(
        lambda: SimpleAvoidanceEnv(config.settings.environment),
        training_config,
    )
    summary = _summary_payload(
        config=config,
        result=result,
        run_id=run_id,
        output_path=output_path,
        history_path=paths.episode_jsonl,
        checkpoint_path=checkpoint_path,
    )
    writer.write_summary(output_path, summary)
    return json.loads(Path(output_path).read_text(encoding="utf-8"))


def _training_config(
    config: HCATrainingRunConfig,
    *,
    checkpoint_path: Path | None = None,
    history_path: Path | None = None,
) -> HCAPPOTrainingConfig:
    total_timesteps = config.total_timesteps or config.settings.run.total_timesteps
    rollout_steps = min(config.settings.ppo.rollout_steps, total_timesteps, 64)
    minibatch_size = min(config.settings.ppo.minibatch_size, rollout_steps)
    update_epochs = min(config.settings.ppo.update_epochs, 2)
    return HCAPPOTrainingConfig(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        minibatch_size=minibatch_size,
        update_epochs=update_epochs,
        clip_range=config.settings.ppo.clip_range,
        gamma=config.settings.ppo.gamma,
        gae_lambda=config.settings.ppo.gae_lambda,
        entropy_coef=config.settings.ppo.entropy_coef,
        value_loss_coef=config.settings.ppo.value_loss_coef,
        max_grad_norm=config.settings.ppo.max_grad_norm,
        seed=config.seed if config.seed is not None else config.settings.run.seed,
        torch_num_threads=1,
        network=HCAActorCriticConfig(embedding_dim=16, hidden_sizes=(8,), dropout=0.0),
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


def _summary_payload(
    *,
    config: HCATrainingRunConfig,
    result: PPOTrainingResult,
    run_id: str,
    output_path: Path,
    history_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    training = asdict(result)
    return {
        "schema_version": 1,
        "record_type": "ppo_hca_training_smoke",
        "stage": config.stage,
        "variant": config.variant,
        "run_id": run_id,
        "training": training,
        "metrics": {
            "success_rate": training["success_rate"],
            "collision_rate": training["collision_rate"],
            "timeout_rate": training["timeout_rate"],
            "average_episode_return": training["average_episode_return"],
        },
        "network": {
            "embedding_dim": 16,
            "target_attention_heads": 4,
            "threat_attention_heads": 4,
            "hidden_sizes": [8],
        },
        "artifacts": {
            "summary_json": str(output_path),
            "training_history_jsonl": str(history_path),
            "checkpoint_path": str(checkpoint_path),
        },
    }


def _config_hash(settings: TrainingSettings, total_timesteps: int, stage: str, variant: str) -> str:
    material = repr((settings, total_timesteps, stage, variant)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()
