# PyBullet Visibility Waypoint Prior Design

## Purpose

The current randomized PyBullet candidate reaches `0.69` success with `0.01`
collision and `0.30` timeout on the 100-scenario development range. Increasing
training from 100,000 to 150,000 steps does not improve those metrics. The
remaining failures are predominantly safe timeouts caused by lateral drift and
local APF minima in multi-obstacle layouts.

This iteration replaces the APF bypass heuristic with a deterministic 2D
visibility-graph waypoint prior while preserving PPO+MLP as the trained policy,
the 27D observation contract, the 3D action contract, all reward coefficients,
and every existing validation and final-holdout threshold.

## Evidence And Boundaries

- Lowering `repulsive_gain` from `0.02` to `0.01` changes safe timeouts into
  collisions: 100 development scenarios produce `0.74` success, `0.13`
  collision, and `0.13` timeout under the APF-only diagnostic.
- Restoring adaptive speed repeats an existing failed experiment. Its 100,000
  step checkpoint uses `min_speed_fraction=0.4` and validates at `0.35` success
  with `0.62` timeout. Its deterministic speed remains almost constant around
  `0.69`; fixed full speed is materially better.
- APF saturation is not the dominant failure. Only `2.58%` of failed-trajectory
  steps have a prior angle too large for the PPO residual to counteract.
- The visibility planner may use only obstacle positions and radii already
  present in the observation. It must not access simulator internals, scenario
  seeds, validation outcomes, or final-holdout layouts.
- Development remains on seeds `500000..500099`. Reserved final seeds
  `2000000..2000099` remain untouched until all candidate gates pass.

## Architecture

Create `swift.rl.visibility_planner` as a Torch-free geometry module. It accepts
the relative goal and extended obstacle slots from an observation and returns a
relative waypoint. The planner treats the vehicle as the origin and operates in
the horizontal plane.

For each obstacle, the collision radius is:

```text
obstacle radius + visibility clearance
```

The randomized configuration uses a visibility clearance of `0.18 m`, covering
the `0.061 m` CF2X collision envelope, the `0.10 m` safety margin, and a small
tracking buffer while remaining inside the sampler's endpoint-clearance
contract.

The graph contains the origin, the goal, and equally spaced nodes around every
inflated obstacle. Ring nodes use radius
`inflated_radius / cos(pi / sample_count)`, so the chord between adjacent nodes
is tangent to the inflated circle rather than cutting through it. Two nodes are
connected when their segment stays outside every inflated obstacle. Dijkstra's
algorithm selects the shortest path with stable node-index tie breaking. A
clear direct path returns the goal unchanged; no-path and malformed optional
slots fall back to the relative goal without raising during control.

`apf_features_from_observation` uses the visibility waypoint as its attraction
target when the planner is enabled. Existing repulsive-force calculation is
unchanged. The PPO action prior and residual composition are unchanged, which
keeps this iteration focused on path selection rather than controller tuning.

## Configuration And Lineage

Add three fields to `APFConfig` and `PyBulletAPFActionPriorSettings`:

- `visibility_planner_enabled: bool = False`
- `visibility_clearance: float = 0.18`
- `visibility_samples: int = 16`

`visibility_clearance` must be positive and `visibility_samples` must be at
least eight. YAML parsing, APF conversion, PPO/HCA checkpoint serialization,
and evaluator runtime metadata must preserve the fields. The randomized
training YAML enables the visibility planner and disables the superseded bypass
heuristic. Existing checkpoints without the fields remain loadable through
dataclass defaults.

## Testing

Geometry unit tests cover a clear direct route, a centered blocker, a
multi-obstacle cluster, deterministic side selection, and collision-free graph
segments. APF tests prove the selected waypoint changes attraction without
changing repulsion. Configuration and checkpoint tests prove round-trip
lineage. Existing 15D observations continue to use the nearest-obstacle slot
and remain valid.

After repository verification, real PyBullet execution follows the existing
gates without threshold changes:

1. 100-layout geometry validation.
2. 2,048-step smoke.
3. 32,768-step pilot with 20 development scenarios and success at least `0.20`.
4. 100,000-step candidate with 100 development scenarios: success at least
   `0.80`, collision at most `0.05`, timeout at most `0.20`.
5. Three 150,000-step seeds only after the candidate passes.
6. One reserved-final evaluation followed by the strict robustness gate:
   worst success at least `0.95`, collision `0.0`, timeout at most `0.05`, and
   worst average minimum safety distance at least `0.10 m`.

## Failure Handling

If the planner candidate fails, diagnostics compare direct-path visibility,
selected waypoint, path length, and failure type on development seeds. Any next
iteration changes one of planner geometry, controller tracking, or policy
training, never all three together. The final holdout is not used for tuning.

