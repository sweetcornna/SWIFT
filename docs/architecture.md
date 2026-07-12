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

## Stabilized-hover boundary

`src/swift/hover/` adds a second, explicitly separate integration surface for
single-drone robust hover. SWIFT owns the immutable action/configuration,
selection, artifact-validation, publication, and command contracts. The external PyBullet repository remains the runtime substrate and owns
`HoverAviary`, Stable-Baselines3 execution, reward/reset wrappers, and raw
trajectory generation. Its location is supplied by `configs/simulation.yaml`
(the portable default is the adjacent `../pybullet` directory). The bridge invokes
substrate scripts with Pixi and does not add the substrate to `sys.path` or vendor
it into this repository.

The `ct_att_yawrate_v1` policy action is `(collective thrust, desired roll,
desired pitch, desired yaw rate)`. A transactional CF2X attitude controller
maps it to motor RPM at 30 Hz while physics runs at 240 Hz. Observations retain
12 physical features and 60 high-level-action history features; generated RPM
is excluded. Training uses `robust_uniform_v1`, `hover_kin_safe_v1`,
`train_rms_physical12_v1`, and a frozen 100-case held-out bank with
`heldout_robust_lexicographic_v1` checkpoint selection: maximize safety completions, minimize p95
final-window altitude MAE, then maximize shaped return, with earliest ties.

This hover integration is additive and does not replace or alter SWIFT's
velocity-action navigation, obstacle curriculum, APF/HCA, or accumulated-heading
behavior. Hover evidence supports only the documented stabilized-hover scope; it is
not navigation or real-flight safety evidence.

## Hover artifact integrity

Every accepted model is bound to an exact observation-normalization sidecar by
SHA-256 in schema-5 `summary.json`. Evaluation validates those bindings and is
read-only with respect to training inputs. The curated publication contains only
seeds 1–3 `robust_best_model.zip` and `final_model.zip`, their exact sidecars,
`action_profile.json`, `source_hashes.json`, training summaries, compact
aggregate evaluation JSON, and a SHA-256 manifest. Raw trajectory NPZ files,
validation banks, callback archives, and the external simulator source are not
published.
