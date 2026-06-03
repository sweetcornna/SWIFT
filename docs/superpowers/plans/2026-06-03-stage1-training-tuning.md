# Stage 1 Training And Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real CPU PPO+MLP training smoke, strict experiment artifacts, checkpoint/history evidence, and deterministic Stage 1 tuning that proves an obstacle collision can be tuned into success.

**Architecture:** Public RL APIs remain Torch-lazy. PyTorch code lives in `src/swift/rl/torch_ppo.py`; experiment runners own artifact output; tuning stays deterministic and repo-native. PyBullet remains command-level smoke only.

**Tech Stack:** Python 3.14, PyYAML, optional `torch>=2.12,<3`, optional `numpy>=2.4,<3`, pytest.

---

## Task 1: PPO Public API And Torch Trainer

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/swift/rl/ppo.py`
- Create: `src/swift/rl/torch_ppo.py`
- Modify: `src/swift/rl/__init__.py`
- Test: `tests/rl/test_ppo_optional.py`
- Test: `tests/rl/test_torch_ppo.py`

- [x] Write failing tests for Torch-lazy imports, actor/value shapes, bounded actions, finite GAE, and one 128-step CPU train call.
- [x] Add optional extra `train = ["numpy>=2.4,<3", "torch>=2.12,<3"]`.
- [x] Add `TorchUnavailableError`, `MLPActorCriticConfig`, `PPOTrainingConfig`, `PPOTrainingResult`, and lazy `train_ppo_mlp(...)`.
- [x] Implement `torch_ppo.py` with MLP actor-critic, diagonal Gaussian sampling, action scaling, rollout collection, GAE, PPO clipped loss, and finite result metrics.
- [x] Persist strict update-history JSONL and a checkpoint containing model, optimizer, and RNG state.
- [x] Load checkpoints back into an MLP actor-critic and evaluate them deterministically.
- [x] Verify: `python -m pytest tests\rl\test_ppo_optional.py tests\rl\test_torch_ppo.py -q`.

## Task 2: Artifact Writer And Evaluation Metrics

**Files:**
- Create: `src/swift/experiments/artifacts.py`
- Modify: `src/swift/experiments/schema.py`
- Modify: `src/swift/experiments/__init__.py`
- Test: `tests/experiments/test_artifacts.py`

- [x] Write failing tests for JSONL writes, strict summary JSON, deterministic run IDs, Windows-safe checkpoint filenames, and artifact paths.
- [x] Implement `ExperimentArtifactConfig`, `ExperimentArtifactPaths`, `ExperimentArtifactWriter`, `build_run_id(...)`, and `checkpoint_filename(...)`.
- [x] Extend metric enum with timeout, average smoothness, minimum safety distance, average return, and mean steps.
- [x] Verify: `python -m pytest tests\experiments\test_artifacts.py -q`.

## Task 3: Training Config And PPO Smoke CLI

**Files:**
- Modify: `configs/training.yaml`
- Modify: `src/swift/config/settings.py`
- Modify: `src/swift/config/__init__.py`
- Create: `src/swift/experiments/ppo_training_runner.py`
- Create: `scripts/run_ppo_mlp_smoke.py`
- Test: `tests/config/test_training_settings.py`
- Test: `tests/experiments/test_ppo_training_runner.py`
- Test: `tests/test_ppo_mlp_smoke_script.py`

- [x] Write failing tests for loading `configs/training.yaml`, rejecting invalid PPO settings, producing JSON-safe training summary, and CLI `--help` / smoke output.
- [x] Add typed training settings for PPO, network, run, environment, and artifact paths.
- [x] Implement runner that builds `SimpleAvoidanceEnv`, calls `train_ppo_mlp`, writes summary JSON, and returns a strict payload.
- [x] Link summary JSON to the training-history JSONL and checkpoint artifacts.
- [x] Implement CLI with `--config`, `--total-timesteps`, `--seed`, `--output`, and `--dry-run`.
- [x] Verify: `python -m pytest tests\config\test_training_settings.py tests\experiments\test_ppo_training_runner.py tests\test_ppo_mlp_smoke_script.py -q`.

## Task 4: Deterministic Stage 1 Tuning

**Files:**
- Modify: `configs/training.yaml`
- Create: `src/swift/experiments/tuning_runner.py`
- Create: `scripts/run_stage1_tuning.py`
- Modify: `src/swift/experiments/__init__.py`
- Test: `tests/experiments/test_tuning_runner.py`
- Test: `tests/test_stage1_tuning_script.py`

- [x] Write failing tests that the default policy collides in the fixed obstacle scenario and grid tuning finds a successful candidate.
- [x] Implement `Stage1Scenario`, `PolicySearchSpace`, `TuningRunConfig`, and `run_stage1_policy_search(...)`.
- [x] Rank candidates by success, collision avoidance, path length, smoothness, safety distance, and steps.
- [x] Implement CLI writing deterministic strict JSON.
- [x] Verify: `python -m pytest tests\experiments\test_tuning_runner.py tests\test_stage1_tuning_script.py -q`.

## Final Verification

- [x] `python -m pytest`
- [x] `python scripts\swift_healthcheck.py`
- [x] `python scripts\run_pybullet_smoke.py --check-only`
- [x] `python scripts\run_pybullet_runtime_smoke.py --steps 1`
- [x] `python scripts\run_ppo_mlp_smoke.py --total-timesteps 128 --output outputs\training\ppo_smoke.json`
- [x] `python scripts\run_ppo_checkpoint_eval.py --checkpoint <checkpoint_path> --episodes 3 --output outputs\evaluation\ppo_checkpoint_eval.json`
- [x] `python scripts\run_stage1_tuning.py --output outputs\tuning\stage1_grid.json`
- [x] `python scripts\run_stage1_report.py --ppo-summary outputs\training\ppo_smoke.json --tuning-summary outputs\tuning\stage1_grid.json --output outputs\reports\stage1_training_tuning_report.json`
- [ ] Confirm `git status -sb` is clean after committing and pushing.

## Self-Review

- Scope is aligned with the user's training/tuning objective while keeping later PyBullet runtime training deferred.
- No task depends on direct PyBullet internals.
- Each worker has a disjoint primary ownership area; shared exports and `configs/training.yaml` require integration review.
- Code quality review fixes are included: PPO YAML coefficients accept zero consistently with `PPOConfig`, and tuning configs reject non-finite values before JSON/ranking.
