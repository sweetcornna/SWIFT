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
substrate smoke until the environment adapter stabilizes.

The current Stage 1 training slice adds an optional Torch CPU PPO smoke and a
deterministic tuning grid. PPO smoke verifies the train/update/artifact loop;
the tuning grid uses a fixed obstacle scenario where the default policy collides
and the selected candidate reaches the goal safely.

The current artifact surface includes strict PPO summary JSON, update-history
JSONL, a Torch checkpoint containing model, optimizer, and RNG state, the
deterministic tuning JSON, and a combined Stage 1 training/tuning report.

## Stage 2: PPO+HCA Upgrade

Replace flat MLP perception with target and threat attention layers. Preserve
the PPO baseline as the comparison target.

## Stage 3: HCA+APF Fusion

Embed APF attraction and repulsion vectors into the HCA perception flow. Measure
convergence speed, success rate, path smoothness, and minimum safety distance.

## Stage 4: Multi-Scenario Evaluation

Run ablations across building density, dynamic drone count, delivery targets,
and altitude preferences. Produce reports, demo video inputs, and final project
evidence.
