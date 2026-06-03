from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json

from swift.config import TrainingSettings
from swift.envs import SimpleAvoidanceEnv
from swift.experiments.artifacts import (
    ExperimentArtifactWriter,
    artifact_reference,
    build_artifact_manifest,
    build_run_id,
    checkpoint_filename,
)
from swift.rl.apf import APFConfig
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
    enable_apf: bool = False

    def __post_init__(self) -> None:
        if self.total_timesteps is not None and self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "stage", str(self.stage).strip() or "stage2")
        variant = str(self.variant).strip() or "ppo_hca"
        if self.enable_apf and variant == "ppo_hca":
            variant = "ppo_hca_apf"
        object.__setattr__(self, "variant", variant)


def run_hca_training_smoke(config: HCATrainingRunConfig) -> dict[str, Any]:
    base_training_config = _training_config(config)
    writer = ExperimentArtifactWriter(config.settings.artifact)
    config_hash = _config_hash(config.settings, base_training_config.total_timesteps, config.stage, config.variant)
    run_id = build_run_id(
        stage=config.stage,
        variant=config.variant,
        seed=base_training_config.seed,
        config_hash=config_hash,
        started_at_utc=datetime.now(UTC),
    )
    paths = writer.paths_for(config.stage, config.variant, run_id)
    output_path = config.output or paths.summary_json
    manifest_path = _manifest_sidecar_path(output_path) if config.output is not None else paths.manifest_json
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
        manifest_path=manifest_path,
        config_hash=config_hash,
    )
    writer.write_summary(output_path, summary)
    writer.write_manifest(
        manifest_path,
        build_artifact_manifest(
            subject_record_type=summary["record_type"],
            run_id=run_id,
            stage=config.stage,
            variant=config.variant,
            lineage=summary["lineage"],
            outputs=[
                artifact_reference(output_path, role="summary_json"),
                artifact_reference(paths.episode_jsonl, role="training_history_jsonl"),
                artifact_reference(checkpoint_path, role="checkpoint"),
            ],
        ),
    )
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
    network_config = HCAActorCriticConfig(
        embedding_dim=16,
        hidden_sizes=(8,),
        dropout=0.0,
        apf_config=APFConfig() if config.enable_apf else None,
    )
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
        network=network_config,
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
    manifest_path: Path,
    config_hash: str,
) -> dict[str, Any]:
    training = asdict(result)
    return {
        "schema_version": 1,
        "record_type": "ppo_hca_training_smoke",
        "stage": config.stage,
        "variant": config.variant,
        "run_id": run_id,
        "lineage": {
            "training_backend": "torch_ppo_hca",
            "config_hash": config_hash,
        },
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
            "apf_enabled": config.enable_apf,
            "apf_config": asdict(APFConfig()) if config.enable_apf else None,
        },
        "artifacts": {
            "summary_json": str(output_path),
            "training_history_jsonl": str(history_path),
            "checkpoint_path": str(checkpoint_path),
            "manifest_json": str(manifest_path),
        },
    }


def _config_hash(settings: TrainingSettings, total_timesteps: int, stage: str, variant: str) -> str:
    material = repr((settings, total_timesteps, stage, variant)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _manifest_sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")
