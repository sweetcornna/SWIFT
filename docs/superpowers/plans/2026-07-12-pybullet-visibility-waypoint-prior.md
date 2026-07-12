# PyBullet Visibility Waypoint Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the local APF bypass heuristic with a deterministic visibility-graph waypoint prior that can route around up to three observed static obstacles while preserving PPO residual control and strict robustness gates.

**Architecture:** Add a Torch-free visibility planner that constructs collision-free graph edges from the existing extended observation, then use its first waypoint as the APF attraction target. Propagate planner settings through YAML, checkpoints, and reports without changing rewards, PPO optimization, action limits, or holdout policy.

**Tech Stack:** Python 3.11, dataclasses, heapq, math, PyTorch PPO, PyBullet, pytest, YAML.

---

### Task 1: Visibility Graph Geometry

**Files:**
- Create: `src/swift/rl/visibility_planner.py`
- Create: `tests/rl/test_visibility_planner.py`

- [ ] **Step 1: Write failing direct-path and blocker tests**

Add tests that call `visibility_waypoint_from_observation` with a 27D observation. Assert a clear route returns the relative goal, a centered blocker returns a waypoint with non-zero lateral displacement, repeated calls return the same waypoint, and the segment to that waypoint stays outside `obstacle_radius + clearance`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/rl/test_visibility_planner.py -q`

Expected: collection fails because `swift.rl.visibility_planner` does not exist.

- [ ] **Step 3: Implement the minimal visibility graph**

Create immutable `VisibilityPlannerConfig(clearance=0.18, samples=16)`,
`visibility_waypoint_from_observation(observation, config)`, obstacle-slot
parsing, sampled ring nodes, segment-to-circle clearance, deterministic
Dijkstra search, and direct-goal fallback. Reject non-positive clearance and
sample counts below eight.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/rl/test_visibility_planner.py -q`

Expected: all visibility-planner tests pass.

### Task 2: APF Integration And Configuration

**Files:**
- Modify: `src/swift/rl/apf.py`
- Modify: `src/swift/config/settings.py`
- Modify: `src/swift/rl/ppo.py`
- Modify: `src/swift/rl/hca.py`
- Modify: `src/swift/rl/torch_hca_ppo.py`
- Modify: `configs/training_pybullet_randomized.yaml`
- Modify: `tests/rl/test_apf_features.py`
- Modify: `tests/config/test_training_settings.py`

- [ ] **Step 1: Add failing APF and settings tests**

Assert `APFConfig` and `PyBulletAPFActionPriorSettings` preserve
`visibility_planner_enabled=True`, `visibility_clearance=0.18`, and
`visibility_samples=16`; assert APF attraction follows a lateral visibility
waypoint for a blocker while repulsion remains independently calculated; assert
the strict randomized YAML enables visibility planning and disables bypass.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/rl/test_apf_features.py tests/config/test_training_settings.py -q`

Expected: failures report missing visibility planner fields.

- [ ] **Step 3: Implement settings propagation**

Add validated fields and mapping conversion to both config dataclasses. In
`_attraction_target`, call `visibility_waypoint_from_observation` before the
legacy bypass path when visibility planning is enabled. Preserve the fields in
all explicit APFConfig reconstruction and checkpoint mapping sites. Enable the
planner in randomized YAML with clearance `0.18` and 16 samples; set
`bypass_enabled: false`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/rl/test_visibility_planner.py tests/rl/test_apf_features.py tests/config/test_training_settings.py -q`

Expected: all selected tests pass.

### Task 3: PPO Checkpoint And Action Contract

**Files:**
- Modify: `tests/rl/test_torch_ppo.py`
- Modify: `tests/rl/test_ppo_optional.py`
- Modify: `tests/experiments/test_pybullet_training_runner.py`

- [ ] **Step 1: Add failing round-trip tests**

Extend the MLP checkpoint and runner tests to assert all three visibility fields
survive serialization and appear in runtime report metadata. Add an action test
whose centered blocker produces a non-zero visibility-guided heading with zero
raw PPO residual.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/rl/test_torch_ppo.py tests/rl/test_ppo_optional.py tests/experiments/test_pybullet_training_runner.py -q`

Expected: the new field and action assertions fail before propagation is complete.

- [ ] **Step 3: Complete minimal propagation fixes**

Update only explicit APF reconstruction or serialization sites exposed by the
failing tests. Do not alter PPO losses, rewards, speed mapping, heading limits,
or network shape.

- [ ] **Step 4: Verify GREEN and repository health**

Run:

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: zero test failures, compile exit `0`, and no whitespace errors.

### Task 4: Staged Real-PyBullet Gates

**Files:**
- Generated: `outputs/evaluation/pybullet_curriculum_visibility_geometry_validation.json`
- Generated: `outputs/training/pybullet_curriculum_visibility_smoke_1x2048.json`
- Generated: `outputs/training/pybullet_curriculum_visibility_pilot_1x32768.json`
- Generated: `outputs/evaluation/pybullet_curriculum_visibility_pilot_validation_20.json`
- Generated after pilot pass: `outputs/training/pybullet_curriculum_visibility_candidate_1x100k.json`
- Generated after pilot pass: `outputs/evaluation/pybullet_curriculum_visibility_candidate_validation_100.json`

- [ ] **Step 1: Run geometry and smoke gates**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_geometry_check.py --seed 500000 --scenarios 100 --output outputs\evaluation\pybullet_curriculum_visibility_geometry_validation.json --fail-on-invalid
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 2048 --output outputs\training\pybullet_curriculum_visibility_smoke_1x2048.json
```

Expected: valid geometry, no first-step collisions, finite smoke metrics, and a checkpoint artifact.

- [ ] **Step 2: Run the short pilot gate**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 32768 --output outputs\training\pybullet_curriculum_visibility_pilot_1x32768.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_visibility_pilot_1x32768.json --episodes 20 --holdout-seed 500000 --output outputs\evaluation\pybullet_curriculum_visibility_pilot_validation_20.json
```

Expected: at least one final-phase training episode and validation success at least `0.20`.

- [ ] **Step 3: Run the 100,000-step candidate gate**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 100000 --output outputs\training\pybullet_curriculum_visibility_candidate_1x100k.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_visibility_candidate_1x100k.json --episodes 100 --holdout-seed 500000 --output outputs\evaluation\pybullet_curriculum_visibility_candidate_validation_100.json
```

Expected: success at least `0.80`, collision at most `0.05`, and timeout at most `0.20`. Stop escalation and diagnose if any metric fails.

### Task 5: Full Candidate And Reserved Final Holdout

**Files:**
- Generated after candidate pass: `outputs/training/pybullet_curriculum_visibility_3x150k.json`
- Generated once after full training: `outputs/evaluation/pybullet_curriculum_visibility_3x150k_final_holdout.json`
- Generated once after final evaluation: `outputs/evaluation/pybullet_curriculum_visibility_3x150k_gate.json`

- [ ] **Step 1: Train three full candidates**

Run:

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 9 10 --total-timesteps 150000 --output outputs\training\pybullet_curriculum_visibility_3x150k.json
```

Expected: all three seeds complete and aggregate timesteps equal `450000`.

- [ ] **Step 2: Consume the reserved holdout once**

Run:

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_visibility_3x150k.json --episodes 100 --holdout-seed 2000000 --output outputs\evaluation\pybullet_curriculum_visibility_3x150k_final_holdout.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_robustness_gate.py --input outputs\evaluation\pybullet_curriculum_visibility_3x150k_final_holdout.json --output outputs\evaluation\pybullet_curriculum_visibility_3x150k_gate.json --fail-on-reject
```

Expected: strict gate exit `0`, worst success at least `0.95`, collision `0.0`, timeout at most `0.05`, and worst average minimum safety distance at least `0.10 m`.

## Execution Record

Implementation and staged training completed on 2026-07-12. The final artifacts
are:

- `outputs/training/pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k.json`
- `outputs/evaluation/pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k_final_holdout.json`
- `outputs/evaluation/pybullet_curriculum_visibility_h1200_rg0002_r025_3x150k_gate.json`

The final report records `robustness_claim=true` and `passed_gates=10/10`.
