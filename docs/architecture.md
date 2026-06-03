<!-- D:\project\SWIFT\docs\architecture.md -->
# Architecture

## System Boundary

SWIFT is the main project repository. `D:\project\pybullet` is an external local
simulation substrate.

## Layers

1. Configuration layer: YAML files define project, simulation, training, and
   evaluation settings.
2. Simulation adapter layer: `PyBulletBackend` validates the local substrate and
   constructs Pixi task commands. `PyBulletVelocityRuntimeEnv` is an optional
   headless runtime adapter for the vendored `VelocityAviary`.
3. Core domain layer: drone states, actions, obstacles, rewards, and metrics.
4. Environment contract layer: Gymnasium-style boundaries for later runtime
   implementation.
5. RL layer: PPO, HCA, and APF configuration contracts plus optional Torch PPO
   MLP and PPO+HCA smoke implementations loaded lazily.
6. Experiment layer: ablation variants, metric naming, strict JSON artifacts,
   update-history JSONL, checkpoints, training smoke runners, deterministic
   checkpoint evaluation, tuning reports, and Stage 1 evidence aggregation.

## Evidence Profiles

`configs/evaluation.yaml` owns evidence profile thresholds consumed by the
convergence gate. Experiment reports emit an evidence level, and gate reports
decide whether that evidence may support a project claim.

- `cpu_smoke_ablation` proves the train/update/artifact path can execute across
  PPO MLP, PPO HCA, and PPO HCA APF variants. cpu_smoke_ablation is not
  convergence evidence.
- `deterministic_multi_scenario_tuning` proves deterministic scenario coverage,
  acceptance ranking, and trajectory/metrics artifact generation.
  deterministic_multi_scenario_tuning is not convergence evidence.
- `long_training_convergence` is the only profile intended for convergence
  claims. For enterprise reporting, only long_training_convergence may support a
  convergence claim, and only when the configured gate passes.

## Integration Rule

The first integration uses command boundaries. Direct Python imports from the
PyBullet substrate are deferred until SWIFT owns a stable environment API.

## Adapter Boundary

The current PyBullet adapter is a command adapter: it validates the external
substrate and builds Pixi task commands for smoke checks. The future environment
adapter will translate SWIFT environment contracts into simulator reset, step,
and observation calls after the Stage 1 repo-native baseline is stable.

The first runtime adapter is deliberately narrow. It uses one drone, `gui=False`,
`record=False`, `user_debug_gui=False`, and maps PyBullet's 20D drone state into
SWIFT's 15D observation contract for runtime smoke evidence. It does not yet own
the final training reward or obstacle semantics.
