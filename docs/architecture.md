<!-- D:\project\SWIFT\docs\architecture.md -->
# Architecture

## System Boundary

SWIFT is the main project repository. `D:\project\pybullet` is an external local
simulation substrate.

## Layers

1. Configuration layer: YAML files define project, simulation, training, and
   evaluation settings.
2. Simulation adapter layer: `PyBulletBackend` validates the local substrate and
   constructs Pixi task commands.
3. Core domain layer: drone states, actions, obstacles, rewards, and metrics.
4. Environment contract layer: Gymnasium-style boundaries for later runtime
   implementation.
5. RL layer: PPO, HCA, and APF configuration contracts plus optional Torch PPO
   training implementation loaded lazily.
6. Experiment layer: ablation variants, metric naming, strict JSON artifacts,
   training smoke runners, and deterministic tuning reports.

## Integration Rule

The first integration uses command boundaries. Direct Python imports from the
PyBullet substrate are deferred until SWIFT owns a stable environment API.

## Adapter Boundary

The current PyBullet adapter is a command adapter: it validates the external
substrate and builds Pixi task commands for smoke checks. The future environment
adapter will translate SWIFT environment contracts into simulator reset, step,
and observation calls after the Stage 1 repo-native baseline is stable.
