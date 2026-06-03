# SWIFT

SWIFT is the enterprise-grade main repository for the Low-Altitude Swift Wing
urban drone delivery 3D obstacle avoidance project.

The repository orchestrates project configuration, simulation integration,
domain contracts, health checks, documentation, and future PPO/HCA/APF
experiments. The local PyBullet substrate remains at `D:\project\pybullet` and
is accessed through an adapter.

## Quick Start

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts\swift_healthcheck.py
python scripts\run_pybullet_smoke.py --check-only
```

### Stage 1 Baseline Smoke

```powershell
python scripts\run_baseline_demo.py --episodes 3 --output outputs\baseline\baseline_metrics.json
```

### Stage 1 Training And Tuning

```powershell
python -m pip install -e ".[dev,train]"
python scripts\run_ppo_mlp_smoke.py --total-timesteps 128 --output outputs\training\ppo_smoke.json
python scripts\run_stage1_tuning.py --output outputs\tuning\stage1_grid.json
python scripts\run_stage1_report.py --ppo-summary outputs\training\ppo_smoke.json --tuning-summary outputs\tuning\stage1_grid.json --output outputs\reports\stage1_training_tuning_report.json
```

The PPO smoke also writes an update-by-update training history JSONL under
`outputs\episodes\...` and a resumable checkpoint under `checkpoints\...`.
Use the `checkpoint_path` recorded in the PPO summary to run deterministic
checkpoint evaluation:

```powershell
python scripts\run_ppo_checkpoint_eval.py --checkpoint <checkpoint_path> --episodes 3 --output outputs\evaluation\ppo_checkpoint_eval.json
```

## Repository Boundary

- SWIFT owns enterprise project structure and experiment orchestration.
- `D:\project\pybullet` owns the existing Pixi-managed PyBullet demos.
- The first integration uses command-level health checks instead of importing
  PyBullet internals directly.
