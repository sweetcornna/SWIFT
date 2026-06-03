# Stage 1 Training And Tuning Design

## Purpose

This slice moves SWIFT beyond a deterministic smoke baseline into a verifiable
training and tuning loop. It adds a CPU-first PPO+MLP trainer for
`SimpleAvoidanceEnv`, strict experiment artifacts, and a deterministic Stage 1
policy tuning scenario that proves an avoidable collision can be converted into
a successful route.

## Boundaries

- `D:\project\pybullet` remains an external command-level substrate.
- Stage 1 training uses only SWIFT's repo-native `SimpleAvoidanceEnv`.
- Torch and NumPy are optional training extras, not base runtime dependencies.
- Public imports from `swift.rl` must not import Torch eagerly.
- Training smoke proves finite updates and artifacts, not final convergence.

## Components

### PPO Trainer

`src/swift/rl/ppo.py` owns public dataclasses and a lazy `train_ppo_mlp(...)`
entry point. `src/swift/rl/torch_ppo.py` owns the real PyTorch implementation:
MLP actor-critic, rollout collection, GAE, clipped PPO updates, and CPU-only
smoke training.

### Experiment Artifacts

`src/swift/experiments/artifacts.py` owns JSONL episode logs, summary JSON, run
IDs, and checkpoint naming. JSON writes use `allow_nan=False` so non-finite
metrics fail early.

### Training Runner And CLI

`src/swift/experiments/ppo_training_runner.py` builds the Stage 1 environment,
runs PPO training, writes summary artifacts, and returns a strict JSON-safe
payload. `scripts/run_ppo_mlp_smoke.py` exposes this as a Windows-friendly CLI.
The runner now writes strict update-history JSONL and a Torch checkpoint with
model, optimizer, and RNG state for audit and resume preparation.

### Checkpoint Evaluation CLI

`src/swift/experiments/ppo_checkpoint_evaluator.py` loads a PPO checkpoint and
runs deterministic mean-action episodes against `SimpleAvoidanceEnv`.
`scripts/run_ppo_checkpoint_eval.py` exposes this as a CLI so checkpoints are
consumable evidence, not only saved files.

### Tuning Runner And CLI

`src/swift/experiments/tuning_runner.py` evaluates the deterministic
`MLPBaselinePolicy` across a fixed obstacle scenario and grid. The default
policy collides in that scenario; the tuned candidate must succeed without
regressing the no-obstacle baseline. `scripts/run_stage1_tuning.py` writes the
full tuning report.

### Stage 1 Report CLI

`src/swift/experiments/stage1_report.py` merges the PPO smoke summary and tuning
JSON into a strict Stage 1 evidence report. `scripts/run_stage1_report.py`
exposes this as a CLI.

## Acceptance Gates

- `python -m pytest` passes.
- `python scripts\swift_healthcheck.py` passes.
- `python scripts\run_pybullet_smoke.py --check-only` passes.
- `python scripts\run_ppo_mlp_smoke.py --total-timesteps 128 --output outputs\training\ppo_smoke.json` exits 0 and writes finite metrics.
- `python scripts\run_ppo_checkpoint_eval.py --checkpoint <checkpoint_path> --episodes 3 --output outputs\evaluation\ppo_checkpoint_eval.json` exits 0 for the checkpoint path recorded in the PPO summary.
- `python scripts\run_stage1_tuning.py --output outputs\tuning\stage1_grid.json` exits 0 and reports collision-to-success improvement.
- `python scripts\run_stage1_report.py --ppo-summary outputs\training\ppo_smoke.json --tuning-summary outputs\tuning\stage1_grid.json --output outputs\reports\stage1_training_tuning_report.json` exits 0 and reports collision-to-success readiness.
- Generated `outputs/` and `checkpoints/` artifacts remain ignored by Git.

## Deferred Work

This slice does not train in PyBullet, does not add HCA/APF networks, and does
not claim final research convergence. It establishes the repeatable training
and tuning infrastructure needed for those later phases.
