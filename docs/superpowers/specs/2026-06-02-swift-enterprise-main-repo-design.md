# SWIFT Enterprise Main Repository Design

## Purpose

SWIFT is the enterprise-grade main repository for the "Low-Altitude Swift Wing"
urban drone delivery obstacle avoidance project. The repository owns project
governance, product architecture, training and evaluation workflows, experiment
traceability, documentation, and integration contracts.

The existing local project at `D:\project\pybullet` remains the simulation
substrate and is accessed through a narrow adapter layer instead of being copied
into SWIFT.

## Source Basis

The design is based on three user-provided source materials:

- Project presentation: an urban drone delivery 3D obstacle avoidance PPTX.
- Preparation document: a DOCX describing prior work, technical route, team
  roles, schedule, and resource readiness.
- Innovation training application: a PDF with formal project fields, budget,
  research content, schedule, expected outcomes, and application rationale.

The design also uses `D:\project\pybullet`, which already contains
Pixi-managed PyBullet, `gym-pybullet-drones`, drone demo tasks, tests, and a
runnable headless drone control demo.

The project technical line is PPO as the decision base, HCA as the perception
core, APF as a safety prior, and PyBullet as the three-dimensional urban
low-altitude simulation and verification platform.

## Scope

SWIFT will be a new Python project repository, not a direct rewrite of
`D:\project\pybullet`.

In scope:

- Initialize SWIFT as the main Git repository.
- Add project metadata, dependency configuration, and quality tooling.
- Define stable configuration files for project, simulation, training,
  evaluation, and experiment paths.
- Implement a local simulation backend adapter that validates and invokes
  `D:\project\pybullet`.
- Create domain models for drone state, action, observation, obstacle, reward
  terms, and metrics.
- Establish Gymnasium-style environment boundaries for future PPO, HCA, and APF
  development.
- Add documentation for project charter, architecture, implementation roadmap,
  risk register, and source-material traceability.
- Add tests around configuration loading, adapter health checks, metric
  calculation, and interface contracts.

Out of scope for the first repository bootstrap:

- Full PPO training implementation.
- HCA model training.
- APF-HCA deep fusion implementation.
- Long-running GPU experiments.
- GUI-heavy demo recording.
- Modifying `D:\project\pybullet` unless an integration blocker is proven and
  approved.

## Recommended Architecture

Use SWIFT as the orchestration and product repository, with
`D:\project\pybullet` integrated as a local backend.

```text
D:\project\SWIFT
|-- README.md
|-- pyproject.toml
|-- .gitignore
|-- configs\
|   |-- project.yaml
|   |-- simulation.yaml
|   |-- training.yaml
|   `-- evaluation.yaml
|-- docs\
|   |-- project-charter.md
|   |-- architecture.md
|   |-- implementation-roadmap.md
|   |-- risk-register.md
|   `-- source-materials.md
|-- scripts\
|   |-- swift_healthcheck.py
|   `-- run_pybullet_smoke.py
|-- src\
|   `-- swift\
|       |-- config\
|       |-- core\
|       |-- sim\
|       |-- envs\
|       |-- rl\
|       |-- experiments\
|       `-- utils\
`-- tests\
    |-- config\
    |-- core\
    |-- sim\
    `-- contracts\
```

## Repository Responsibilities

`configs` stores human-readable YAML configuration. `simulation.yaml` includes
the default `pybullet_root: D:\project\pybullet` and command settings for
Pixi-backed health checks. `training.yaml` captures PPO defaults without
forcing training to exist in the bootstrap.

`src\swift\config` loads, validates, and resolves project configuration. It
must fail fast when paths are missing or invalid.

`src\swift\core` contains stable domain types: `DroneState`, `DroneAction`,
`ObservationVector`, `ObstacleState`, `RewardBreakdown`, `EpisodeMetrics`, and
metric functions. These types are backend-agnostic.

`src\swift\sim` contains the simulation backend boundary. The first
implementation is `PyBulletBackend`, which checks whether
`D:\project\pybullet` exists, whether `.tools\pixi\pixi.exe` exists, whether
`pixi.toml` declares expected tasks, and whether a smoke command can run.

`src\swift\envs` contains the future Gymnasium-style environment contract. The
bootstrap only defines interfaces and stubs with explicit
unsupported-operation errors because full RL behavior is outside bootstrap
scope.

`src\swift\rl` contains future PPO, HCA, and APF module boundaries. The
bootstrap defines package structure and documentation, not production training
code.

`src\swift\experiments` will hold experiment orchestration and result schemas.
The first version defines metric naming conventions and artifact layout.

`docs` turns the provided materials into enterprise project assets: charter,
architecture, roadmap, risk register, and source traceability.

## PyBullet Integration Contract

The adapter treats `D:\project\pybullet` as an external local dependency.

Required checks:

- `D:\project\pybullet` exists.
- `D:\project\pybullet\pixi.toml` exists.
- `D:\project\pybullet\.tools\pixi\pixi.exe` exists.
- `pixi.toml` exposes `test` and `drone-demo` tasks.
- `.\.tools\pixi\pixi.exe run test` exits successfully.
- Optional smoke check: `.\.tools\pixi\pixi.exe run drone-demo` exits
  successfully in headless mode.

The first integration does not import pybullet internals directly. It uses
command boundaries so SWIFT can verify the substrate without coupling to
experimental script internals. Future phases can add a Python package-level
bridge once the environment API is stable.

## Enterprise Standards

The repository should be usable by a new contributor without reading the source
materials first.

Required standards:

- Single command health check.
- Tests for every bootstrap behavior.
- Configuration-first paths and commands.
- No hard-coded experiment outputs inside source code.
- Clear separation between project governance, simulation adapter, RL model
  code, and experiment artifacts.
- All generated experiment outputs go under an ignored `outputs\` directory.
- Documentation states what is implemented, what is planned, and how to verify
  each stage.

## Initial Quality Gates

The bootstrap is complete only when these checks pass:

- `python -m pytest`
- `python scripts\swift_healthcheck.py`
- `python scripts\run_pybullet_smoke.py --check-only`

The pybullet command smoke test should be optional because it may take longer
and depends on the existing Pixi environment. The health check must report the
exact command to run for a full smoke test.

## Implementation Stages

### Stage 0: Repository Bootstrap

Create the repository skeleton, project metadata, configuration files, docs,
and tests. Verify that SWIFT can locate and validate `D:\project\pybullet`.

Deliverables:

- Git repository initialized.
- Python package skeleton under `src\swift`.
- Config files under `configs`.
- Project docs under `docs`.
- Passing bootstrap tests.

### Stage 1: Simulation Adapter

Implement `PyBulletBackend` with health-check and command execution support.
Keep the adapter small and auditable.

Deliverables:

- Adapter health model.
- Command execution wrapper with timeout and captured output.
- Tests that use temporary fake pybullet roots for deterministic validation.
- Optional live smoke command against `D:\project\pybullet`.

### Stage 2: Domain and Environment Contracts

Define backend-agnostic state, action, observation, reward, and metrics
contracts. Add a Gymnasium-style interface boundary without claiming full
training support.

Deliverables:

- Domain dataclasses.
- Metric functions for success, collision, path length, smoothness, and
  minimum safety distance.
- Contract tests for shape, validation, and metric behavior.

### Stage 3: Research Implementation Roadmap

Prepare the future PPO+MLP baseline, PPO+HCA, and HCA+APF implementation plan.
This stage is documentation and test scaffolding only until the bootstrap is
stable.

Deliverables:

- Implementation roadmap aligned with the May 2026 to May 2027 project
  schedule.
- Experiment matrix for MLP, HCA, and HCA+APF.
- Risk register covering sparse reward, PPO-HCA coupling instability, APF local
  minima, GPU availability, and demo reproducibility.

## Risk Controls

The project uses a baseline-first strategy: build the safe PPO+MLP baseline,
upgrade to HCA, then fuse APF after the baseline and HCA-only paths are
measurable.

- If HCA integration is unstable, PPO+MLP remains the demonstrable baseline.
- If APF fusion introduces local-minimum bias, ablation results should preserve
  HCA-only and MLP baselines.
- If direct Python integration with pybullet is brittle, command-level adapter
  remains the stable fallback.
- If GPU resources are unavailable, smoke tests and deterministic small
  scenarios remain runnable on CPU.
- If source materials and code drift, `docs\source-materials.md` records the
  source file paths and extracted project claims.

## Open Decisions

These are intentionally deferred until after bootstrap verification:

- Whether SWIFT should vendor a pinned copy of `gym-pybullet-drones`.
- Whether `D:\project\pybullet` should expose a formal Python API for SWIFT.
- Exact PPO implementation source: custom PyTorch PPO or a stable RL library.
- Experiment tracking stack: plain JSON/CSV first, TensorBoard next, MLflow
  only if needed.
- Whether training artifacts should be stored locally only or prepared for cloud
  GPU runs.

## Approval State

The user approved the main repository approach on 2026-06-02:

> Build `D:\project\SWIFT` as the new enterprise main repository and integrate
> `D:\project\pybullet` as the local simulation substrate.
