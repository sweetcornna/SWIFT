# PyBullet Curriculum Training Design

## Purpose

This slice turns the failed randomized-obstacle run into a staged, auditable
training pipeline. It fixes invalid obstacle layouts, restores a useful goal
reaching signal, introduces a deterministic obstacle curriculum, and uses
successive validation gates before spending compute on a full multi-seed run.

The final objective remains the existing strict PyBullet robustness gate:

- worst checkpoint success rate at least `0.95`;
- maximum checkpoint collision rate `0.0`;
- maximum checkpoint timeout rate at most `0.05`;
- worst average minimum safety distance at least `0.10 m`.

## Evidence And Root Cause

The existing `3 x 100,000` run produced no successes. Collision rates fell as
training progressed while timeout rates rose, so PPO learned to avoid the
large collision penalty without learning to reach the goal.

Two independent causes were reproduced:

1. The sampler validates an obstacle against a point-sized start position. It
   omits the CF2X collision envelope, measured at approximately `0.061 m`.
   Adding that envelope to the geometric check predicts all `29/29` first-step
   collisions in the previous 100-scenario holdout, with no false positives or
   false negatives.
2. Goal progress contributes at most about `+1` over an episode, while a
   collision contributes `-100`. A timed-out policy therefore receives about
   `-1` to `-3` and is strongly preferred over exploratory collisions. The
   seed-10 checkpoint moved away from the goal even in an obstacle-free replay,
   increasing goal distance from `0.5 m` to `0.883 m`.

The previously solved fixed-obstacle configuration does not establish obstacle
avoidance: its sphere is outside the direct-path safety corridor. It only
establishes that the PPO and PyBullet stack can learn basic goal reaching.

## Boundaries

- Keep the current 15D observation and 3D action contracts.
- Keep accumulated heading commands and commanded-heading observation feedback.
- Keep PPO+MLP as the policy family for this iteration.
- Do not warm-start from the failed randomized checkpoints.
- Do not weaken the existing strict robustness thresholds.
- Do not tune against the consumed `1000000..1000099` holdout scenarios.
- Use `500000..500099` only as a development validation range.
- Reserve `2000000..2000099` for one final holdout evaluation.
- Treat multi-obstacle observation expansion as deferred work unless the staged
  single-blocker pilot proves the current observation insufficient.

## Component Ownership

- `swift.config` owns validated reward, vehicle-geometry, and curriculum
  dataclasses plus YAML parsing.
- `swift.envs.pybullet_obstacle_randomization` owns deterministic phase-aware
  layout sampling and geometry checks. It has no PPO dependency.
- `PyBulletVelocityTrainingEnv` owns phase snapshots, reward calculation, and
  terminal episode diagnostics. It does not decide global training progress.
- The Torch PPO trainer owns timestep progress propagation and generic
  per-phase metric aggregation through optional environment metadata.
- PyBullet training and evaluation runners own artifact lineage, validation
  seed ranges, and conversion of internal results into strict JSON reports.
- Existing multi-seed CLIs remain the expensive-run entry points. Pilot gate
  decisions are explicit CLI/report steps rather than hidden automatic retries.

## Configuration

### Obstacle Geometry

`PyBulletObstacleRandomizationSettings` gains an explicit vehicle collision
radius. The randomized training configuration sets it to `0.061 m` for CF2X.
Endpoint and direct-path checks use the combined geometry:

```text
obstacle radius + vehicle radius + safety margin + configured clearance
```

The start and goal centers must both satisfy this clearance. A required path
blocker uses the obstacle radius, vehicle radius, and safety margin when testing
intersection with the direct route. The same seed and phase must always produce
the same layout.

### Reward Contract

A dedicated PyBullet reward configuration replaces hard-coded coefficients:

| Term | Value | Behavior |
| --- | ---: | --- |
| arrival | `+100` | Applied once on goal termination |
| normalized goal progress | `20x` | Positive toward the goal, negative away |
| collision | `-100` | Applied once on collision termination |
| timeout | `-20` | Applied once on timeout truncation |
| time budget | `-1 / max_steps` | Totals approximately `-1` at the horizon |
| heading smoothness | `-0.05 * abs(delta)` | Preserves the current steering regularizer |

This gives a successful route a return around `+110`, a stationary timeout a
return around `-21`, and a route that moves away from the goal a still lower
return. Collision remains the worst terminal outcome, preserving the safety
preference.

All reward values must be finite. Terminal terms are mutually exclusive, and
the serialized `RewardBreakdown` must sum exactly to the scalar reward returned
by the environment.

### Curriculum

The training configuration defines four ordered phases:

| Progress | Obstacles | Path blocker required |
| --- | ---: | --- |
| `[0.00, 0.20)` | `0` | no |
| `[0.20, 0.40)` | `1` | no |
| `[0.40, 0.70)` | `1` | yes |
| `[0.70, 1.00]` | `1..3` | yes |

Progress is based on completed environment timesteps divided by planned total
timesteps. The PPO trainer informs curriculum-capable environments of progress
before the initial reset and before each later episode reset. A phase never
changes during an active episode.

The environment defaults to progress `1.0`. Checkpoint evaluation therefore
always selects the final phase without relying on evaluator call order.
Curriculum definitions reject empty phase lists, non-monotonic boundaries,
gaps, overlaps, invalid obstacle counts, and a final boundary other than `1.0`.

## Data Flow

1. The multi-seed runner loads reward, geometry, and curriculum settings and
   records them in each run's config hash and artifact lineage.
2. PPO sets the current training progress before reset.
3. The environment snapshots the active phase and derives the layout from the
   episode scenario seed.
4. The sampler validates the full vehicle-obstacle geometry and installs the
   layout in the PyBullet runtime.
5. Each step reports the active curriculum phase, progress snapshot, reward
   breakdown, goal distance, and existing safety information.
6. Terminal episode metrics are aggregated by curriculum phase as well as for
   the whole run.
7. Deterministic checkpoint evaluation uses the final phase and a caller-owned
   seed range.

Training summaries and update-history records include the active/observed
phases. Final reports include per-phase episode counts, success, collision,
timeout, and average return so easy-phase performance cannot mask final-phase
failure. Evaluation episode records add initial, minimum, and final goal
distance plus final position. These compact fields distinguish collision,
circling, and goal-avoidance failures without storing full 600-step trajectories.

## Failure Handling

- Sampling exhaustion raises an error containing the scenario seed, phase,
  obstacle count range, and required endpoint clearance.
- A reset fails before stepping if randomized training has no scenario seed.
- Non-finite reward coefficients, rewards, or report metrics fail immediately.
- Evaluation rejects any configuration that does not resolve to the final
  curriculum phase.
- Pilot gate failure stops escalation. It does not trigger extra full runs or
  silent threshold changes.
- Artifacts record validation and holdout seed ranges so accidental reuse is
  visible in review.

## Verification And Training Gates

Implementation follows test-driven development. Unit and contract tests cover:

- vehicle-radius-aware endpoint and blocker geometry;
- deterministic layouts for the same seed and phase;
- curriculum boundary selection and final-phase evaluation defaults;
- reward algebra for success, collision, timeout, and moving away;
- PPO progress propagation without affecting environments that lack a
  curriculum hook;
- per-phase metrics and JSON-safe artifact lineage;
- CLI configuration and dry-run behavior.

Real PyBullet verification then proceeds in increasing cost order:

1. Runtime geometry check: 100 validation layouts must reset with initial
   minimum safety distance strictly above `0.10 m`; no first-step collisions are
   allowed under a neutral action.
2. Smoke run: one seed at `2,048` steps must complete with finite updates and
   valid curriculum/reward artifacts.
3. Short pilot: one seed at `32,768` steps must produce at least one successful
   final-phase episode and at least `0.20` success on 20 validation scenarios.
4. Candidate pilot: one seed at `100,000` steps must reach at least `0.80`
   success, at most `0.05` collision, and at most `0.20` timeout on all 100
   development validation scenarios.
5. Full candidate: train seeds `8`, `9`, and `10` for `150,000` steps each only
   after the candidate pilot passes.
6. Final holdout: evaluate every final checkpoint once on scenarios
   `2000000..2000099`, then run the unchanged strict robustness gate.

If a pilot fails, use its phase metrics and compact episode diagnostics to form
one new hypothesis. A targeted one-off replay may collect a full trajectory for
that hypothesis. Do not combine reward, observation, network, and optimizer
changes in a single retry.

## Acceptance Criteria

- All repository tests, compile checks, and diff checks pass.
- Real PyBullet randomized reset and training smoke checks pass.
- The geometry check reports zero initially invalid validation layouts.
- Training and evaluation artifacts contain reward, curriculum, geometry, and
  seed-range lineage.
- The staged pilot gates control whether larger runs execute.
- Final robustness is claimed only if the existing strict gate passes on the
  new reserved holdout.

## Deferred Work

- Top-k or attention-based multi-obstacle observations.
- Recurrent policies and larger actor-critic networks.
- Dynamic obstacle curricula.
- Real-flight safety claims; simulator robustness is not flight evidence.
