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
5. RL bootstrap layer: PPO, HCA, and APF configuration contracts.
6. Experiment layer: ablation variants and metric naming.

## Integration Rule

The first integration uses command boundaries. Direct Python imports from the
PyBullet substrate are deferred until SWIFT owns a stable environment API.
