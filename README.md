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
python scripts\run_pybullet_runtime_smoke.py --steps 1
```

### Stage 1 Baseline Smoke

```powershell
python scripts\run_baseline_demo.py --episodes 3 --output outputs\baseline\baseline_metrics.json
```

### Stage 1 Training And Tuning

```powershell
python -m pip install -e ".[dev,train]"
python scripts\run_ppo_mlp_smoke.py --total-timesteps 128 --output outputs\training\ppo_smoke.json
python scripts\run_ppo_hca_smoke.py --total-timesteps 64 --output outputs\training\ppo_hca_smoke.json
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

### PyBullet-Backed PPO Training

Use the local PyBullet substrate at `D:\project\pybullet` through the
`PyBulletVelocityTrainingEnv` adapter. On this Windows setup, the current
project Python can run Torch but cannot import PyBullet directly, while the
PyBullet Pixi environment has the simulator stack but no Torch. The supported
local training environment is therefore an external venv created from the Pixi
Python with system site packages, then extended with SWIFT and Torch:

```powershell
D:\project\pybullet\.pixi\envs\default\python.exe -m venv --system-site-packages D:\project\.venvs\swift-pybullet-pixi
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe -m pip install --upgrade pip
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe -m pip install -e D:\project\SWIFT torch PyYAML
```

The runtime adapter automatically registers the Pixi DLL search directories on
Windows before importing `VelocityAviary`. A 32-step pilot writes a JSON summary,
training-history JSONL, checkpoint, and manifest:

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --total-timesteps 32 --seed 0 --output outputs\training\pybullet_pilot_32.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --total-timesteps 32 --seed 1 --enable-obstacles --output outputs\training\pybullet_obstacles_pilot_32.json
```

The tuned PyBullet PPO curriculum uses explicit training configs that keep the
physics horizon reachable and apply a minimum cruise-speed prior for the
high-frequency velocity-control substrate:

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --training-config configs\training_pybullet_probe.yaml --total-timesteps 32768 --seed 6 --output outputs\training\pybullet_probe_speedprior_32768.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_ppo_training.py --training-config configs\training_pybullet_obstacles.yaml --total-timesteps 32768 --seed 7 --enable-obstacles --output outputs\training\pybullet_obstacles_speedprior_32768.json
```

Use the `checkpoint_path` recorded in each PyBullet training report to run
deterministic PyBullet checkpoint evaluation:

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_checkpoint_eval.py --training-config configs\training_pybullet_probe.yaml --checkpoint <checkpoint_path> --episodes 5 --output outputs\evaluation\pybullet_probe_speedprior_32768_eval.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_checkpoint_eval.py --training-config configs\training_pybullet_obstacles.yaml --checkpoint <checkpoint_path> --episodes 5 --enable-obstacles --output outputs\evaluation\pybullet_obstacles_speedprior_32768_eval.json
```

Successful training reports use `record_type=pybullet_ppo_training_report` and
`training_backend=torch_ppo_mlp_pybullet_velocity`; successful deterministic
evaluation reports use `record_type=pybullet_ppo_checkpoint_evaluation`. Both
report types include `runtime_contract=pybullet_velocity_training_compatibility`
and a strict manifest sidecar with artifact checksums. Omit `--output` when you
want the summary and manifest to use run-id artifact paths instead of a fixed
report path.

APF feature generation is available through `swift.rl.apf_features_from_observation(...)`
as a 9D attraction/repulsion/combined-force vector for later HCA+APF fusion.
PPO+HCA smoke training is available through `scripts\run_ppo_hca_smoke.py` and
uses a Torch-lazy HCA actor-critic over the same 15D observation contract.

## Repository Boundary

- SWIFT owns enterprise project structure and experiment orchestration.
- `D:\project\pybullet` owns the existing Pixi-managed PyBullet demos.
- The first integration uses command-level health checks instead of importing
  PyBullet internals directly.
- The optional runtime adapter wraps the local vendored `VelocityAviary` in
  headless mode. On Python versions without a ready `pybullet` wheel, the
  runtime smoke falls back to the PyBullet Pixi environment.
