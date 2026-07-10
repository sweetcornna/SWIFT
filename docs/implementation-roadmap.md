<!-- D:\project\SWIFT\docs\implementation-roadmap.md -->
# Implementation Roadmap

## Stage 0: Repository Bootstrap

Create package metadata, configuration, source package structure, health checks,
domain contracts, documentation, and tests.

## Stage 1: PPO+MLP Baseline

Build the baseline environment loop and PPO+MLP policy path. This is the safe
demonstrable baseline.

Stage 1 is repo-native first: the baseline smoke exercises SWIFT-owned
environment, policy, and experiment contracts before training is coupled to any
external simulator internals. `D:\project\pybullet` remains command-level
substrate smoke until the environment adapter stabilizes. A first optional
runtime smoke adapter now wraps the local vendored `VelocityAviary` headlessly
and maps its raw 20D drone state into SWIFT's 15D observation shape.

The current Stage 1 training slice adds an optional Torch CPU PPO smoke and a
deterministic tuning grid. PPO smoke verifies the train/update/artifact loop;
the tuning grid uses a fixed obstacle scenario where the default policy collides
and the selected candidate reaches the goal safely.

The current artifact surface includes strict PPO summary JSON, update-history
JSONL, a Torch checkpoint containing model, optimizer, and RNG state, the
deterministic checkpoint evaluation JSON, the deterministic tuning JSON, and a
combined Stage 1 training/tuning report.

## Stage 2: PPO+HCA Upgrade

Replace flat MLP perception with target and threat attention layers. Preserve
the PPO baseline as the comparison target.
The current Stage 2 slice adds a Torch-lazy HCA observation adapter, target and
threat attention feature extractor, HCA actor-critic, and a CPU PPO+HCA smoke
runner with JSON history and checkpoint artifacts.

The ablation runner labels short local variant comparisons as
`cpu_smoke_ablation`. This profile is used to verify implementation health and
artifact lineage only; cpu_smoke_ablation is not convergence evidence.

## Stage 3: HCA+APF Fusion

Embed APF attraction and repulsion vectors into the HCA perception flow. Measure
convergence speed, success rate, path smoothness, and minimum safety distance.
The current APF slice provides deterministic 9D attraction, repulsion, and
combined-force features derived from the existing 15D observation contract,
without changing PPO input dimensions.

## Stage 4: Multi-Scenario Evaluation

Run ablations across building density, dynamic drone count, delivery targets,
and altitude preferences. Produce reports, demo video inputs, and final project
evidence.

The deterministic Stage 4 evaluator emits `deterministic_multi_scenario_tuning`
evidence. It checks scenario coverage and ranked candidate acceptance, but
deterministic_multi_scenario_tuning is not convergence evidence. Long-horizon
training must be recorded separately as `long_training_convergence`; only
long_training_convergence may support a convergence claim after the configured
gate accepts the report.

The PyBullet robustness slice adds deterministic per-episode static-obstacle
randomization, three-seed PPO training, shared-layout checkpoint holdout
evaluation, and a simulator-specific worst-case gate. This evidence remains
single-drone, headless simulation evidence and does not establish real-flight
safety.

The curriculum-training follow-up corrects randomized spawn geometry with the
CF2X collision envelope, scales goal progress, adds an explicit timeout cost,
and moves from goal reaching through single blockers to the final 1-3 obstacle
distribution. A real-runtime geometry gate and successive 2,048, 32,768, and
100,000-step pilots must pass before the 3 x 150,000-step candidate is allowed.
Development validation uses seeds `500000..500099`; the consumed
`1000000..1000099` range is excluded from tuning, and `2000000..2000099` is
reserved for final evidence.
