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

## Repository Boundary

- SWIFT owns enterprise project structure and experiment orchestration.
- `D:\project\pybullet` owns the existing Pixi-managed PyBullet demos.
- The first integration uses command-level health checks instead of importing
  PyBullet internals directly.
