# PyBullet Curriculum Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a deterministic PyBullet obstacle curriculum that removes invalid spawn layouts and teaches PPO to reach the goal before escalating to the strict randomized robustness run.

**Architecture:** Typed configuration owns the reward, vehicle geometry, and ordered curriculum. The PyBullet environment snapshots a phase on reset, while the generic PPO loop only propagates timestep progress and aggregates optional phase metadata. Evaluation always defaults to the final phase and emits compact goal-distance diagnostics; a dedicated geometry validator blocks training when any layout starts inside the safety margin.

**Tech Stack:** Python 3.11/3.14, dataclasses, PyTorch PPO, PyBullet/gym-pybullet-drones, PyYAML, pytest, PowerShell CLIs.

---

## File Map

- `src/swift/config/settings.py`: reward, curriculum phase, curriculum, and vehicle-radius configuration validation.
- `src/swift/config/__init__.py`: public exports for the new configuration types.
- `configs/training_pybullet_randomized.yaml`: CF2X geometry, reward coefficients, four curriculum phases, and 150k full-run default.
- `src/swift/envs/pybullet_obstacle_randomization.py`: deterministic full-vehicle geometry checks and actionable sampling errors.
- `src/swift/envs/pybullet_velocity.py`: phase snapshots, phase-aware layouts, configured reward calculation, and curriculum metadata.
- `src/swift/rl/ppo.py`: immutable per-phase training metric result types.
- `src/swift/rl/__init__.py`: public export for per-phase result types.
- `src/swift/rl/torch_ppo.py`: optional curriculum progress hook and per-phase aggregation.
- `src/swift/experiments/pybullet_training_runner.py`: pass settings into the environment and serialize lineage.
- `src/swift/experiments/pybullet_checkpoint_evaluator.py`: final-phase enforcement and compact episode diagnostics.
- `src/swift/experiments/pybullet_geometry_validator.py`: real-runtime initial-clearance and first-step validation report.
- `src/swift/experiments/__init__.py`: lazy/public exports for geometry validation.
- `scripts/run_pybullet_geometry_check.py`: geometry validation CLI.
- `scripts/run_pybullet_multi_seed_training.py`: honor YAML timestep defaults and 150k artifact naming.
- `scripts/run_pybullet_multi_seed_checkpoint_eval.py`: use the reserved final holdout only when explicitly requested.
- `tests/config/test_training_settings.py`: typed configuration and invalid-boundary contracts.
- `tests/envs/test_pybullet_obstacle_randomization.py`: vehicle-aware sampling contracts.
- `tests/sim/test_pybullet_runtime_adapter.py`: reward, phase selection, terminal precedence, and environment metadata.
- `tests/rl/test_torch_ppo.py`: progress propagation and phase aggregation.
- `tests/experiments/test_pybullet_training_runner.py`: training lineage.
- `tests/experiments/test_pybullet_checkpoint_evaluator.py`: final-phase diagnostics.
- `tests/experiments/test_pybullet_geometry_validator.py`: geometry report behavior.
- `tests/test_pybullet_randomized_training_scripts.py`: CLI dry-run and failure exit behavior.
- `README.md`, `docs/implementation-roadmap.md`: staged workflow, seed hygiene, and evidence boundaries.

## Deliberately Deferred

This plan does not add top-k multi-obstacle observations, attention, recurrence,
larger networks, dynamic obstacles, or real-flight claims. Those changes require
new evidence and a separate design if the current single-blocker pilot proves
the 15D observation contract insufficient.

### Task 1: Preserve The Verified Randomized-Training Baseline

**Files:**
- Commit existing changes in `README.md`, `docs/implementation-roadmap.md`, `configs/training_pybullet_randomized.yaml`, `scripts/run_pybullet_multi_seed_training.py`, `scripts/run_pybullet_multi_seed_checkpoint_eval.py`, `scripts/run_pybullet_robustness_gate.py`, `src/swift/config/`, `src/swift/envs/`, `src/swift/experiments/`, `src/swift/sim/`, and their current tests.

- [ ] **Step 1: Verify the current baseline before staging it**

Run:

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: `238 passed`, compile exits `0`, and `git diff --check` prints nothing. If the test count changes because the repository changed after this plan was written, require zero failures and record the new count in the commit message notes.

- [ ] **Step 2: Confirm only the known randomized-training baseline is uncommitted**

Run:

```powershell
git status --short
git diff --stat
```

Expected: the source, config, scripts, docs, and tests listed in this task; no generated `outputs/` or `checkpoints/` files.

- [ ] **Step 3: Commit the baseline without the implementation-plan file**

```powershell
git add README.md docs/implementation-roadmap.md configs/training_pybullet_randomized.yaml scripts/run_pybullet_multi_seed_training.py scripts/run_pybullet_multi_seed_checkpoint_eval.py scripts/run_pybullet_robustness_gate.py src/swift/config src/swift/envs src/swift/experiments src/swift/sim tests
git commit -m "feat: add randomized pybullet robustness training"
```

Expected: one commit containing the already verified heading fix, deterministic randomized layouts, multi-seed runner, holdout evaluation, robustness gate, and their tests.

### Task 2: Add Typed Reward, Vehicle Geometry, And Curriculum Configuration

**Files:**
- Modify: `src/swift/config/settings.py:110-208`
- Modify: `src/swift/config/__init__.py`
- Modify: `configs/training_pybullet_randomized.yaml`
- Test: `tests/config/test_training_settings.py:163-271`

- [ ] **Step 1: Write failing parser and validation tests**

Add tests that load this complete contract and assert every typed value:

```python
def test_load_training_settings_parses_pybullet_reward_and_curriculum(tmp_path: Path) -> None:
    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        """
pybullet_obstacle_randomization:
  enabled: true
  vehicle_radius: 0.061
pybullet_reward:
  arrival_reward: 100.0
  approach_scale: 20.0
  collision_penalty: 100.0
  timeout_penalty: 20.0
  episode_time_penalty: 1.0
  heading_smoothness_penalty: 0.05
pybullet_curriculum:
  enabled: true
  phases:
    - {name: goal_reaching, end_fraction: 0.20, min_obstacles: 0, max_obstacles: 0, require_path_blocker: false}
    - {name: single_obstacle, end_fraction: 0.40, min_obstacles: 1, max_obstacles: 1, require_path_blocker: false}
    - {name: single_blocker, end_fraction: 0.70, min_obstacles: 1, max_obstacles: 1, require_path_blocker: true}
    - {name: randomized_final, end_fraction: 1.00, min_obstacles: 1, max_obstacles: 3, require_path_blocker: true}
""".strip(),
        encoding="utf-8",
    )

    settings = load_training_settings(config_path)

    assert settings.pybullet_obstacle_randomization.vehicle_radius == pytest.approx(0.061)
    assert settings.pybullet_reward.approach_scale == pytest.approx(20.0)
    assert settings.pybullet_reward.timeout_penalty == pytest.approx(20.0)
    assert [phase.name for phase in settings.pybullet_curriculum.phases] == [
        "goal_reaching",
        "single_obstacle",
        "single_blocker",
        "randomized_final",
    ]
    assert settings.pybullet_curriculum.phase_for(0.0).name == "goal_reaching"
    assert settings.pybullet_curriculum.phase_for(0.20).name == "single_obstacle"
    assert settings.pybullet_curriculum.phase_for(0.70).name == "randomized_final"
    assert settings.pybullet_curriculum.phase_for(1.0).name == "randomized_final"
```

Add parameterized invalid cases with exact messages:

```python
@pytest.mark.parametrize(
    ("yaml_body", "message"),
    [
        ("phases: []", "phases must not be empty"),
        (
            "phases:\n    - {name: first, end_fraction: 0.7, min_obstacles: 0, max_obstacles: 0}\n"
            "    - {name: second, end_fraction: 0.6, min_obstacles: 1, max_obstacles: 1}",
            "end_fraction values must be strictly increasing",
        ),
        (
            "phases:\n    - {name: final, end_fraction: 0.9, min_obstacles: 1, max_obstacles: 1}",
            "final curriculum end_fraction must equal 1.0",
        ),
        (
            "phases:\n    - {name: empty, end_fraction: 1.0, min_obstacles: 0, max_obstacles: 0, require_path_blocker: true}",
            "path blocker requires at least one obstacle",
        ),
    ],
)
def test_load_training_settings_rejects_invalid_pybullet_curriculum(
    tmp_path: Path, yaml_body: str, message: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("pybullet_curriculum:\n  enabled: true\n  " + yaml_body + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_training_settings(path)
```

Add an explicit finite-value contract for rewards:

```python
@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "-1.0"])
def test_load_training_settings_rejects_invalid_pybullet_reward(
    tmp_path: Path, value: str
) -> None:
    path = tmp_path / "invalid-reward.yaml"
    path.write_text(
        f"pybullet_reward:\n  approach_scale: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="approach_scale must be non-negative"):
        load_training_settings(path)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
python -m pytest tests/config/test_training_settings.py -q
```

Expected: failures because `vehicle_radius`, `PyBulletRewardSettings`, and curriculum types do not exist.

- [ ] **Step 3: Implement minimal immutable configuration types**

Add these public dataclasses and parsing paths in `settings.py`:

```python
@dataclass(frozen=True)
class PyBulletRewardSettings:
    arrival_reward: float = 100.0
    approach_scale: float = 1.0
    collision_penalty: float = 100.0
    timeout_penalty: float = 0.0
    episode_time_penalty: float = 1.0
    heading_smoothness_penalty: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "arrival_reward",
            "approach_scale",
            "collision_penalty",
            "timeout_penalty",
            "episode_time_penalty",
            "heading_smoothness_penalty",
        ):
            object.__setattr__(self, name, _non_negative_float(name, getattr(self, name)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PyBulletRewardSettings":
        defaults = cls()
        return cls(**{name: mapping.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__})


@dataclass(frozen=True)
class PyBulletCurriculumPhaseSettings:
    name: str
    end_fraction: float
    min_obstacles: int
    max_obstacles: int
    require_path_blocker: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("curriculum phase name must not be empty")
        end_fraction = _unit_interval("end_fraction", self.end_fraction)
        min_obstacles = _non_negative_int("min_obstacles", self.min_obstacles)
        max_obstacles = _non_negative_int("max_obstacles", self.max_obstacles)
        if max_obstacles < min_obstacles:
            raise ValueError("max_obstacles must be >= min_obstacles")
        blocker = _bool_value("require_path_blocker", self.require_path_blocker)
        if blocker and max_obstacles == 0:
            raise ValueError("path blocker requires at least one obstacle")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "end_fraction", end_fraction)
        object.__setattr__(self, "min_obstacles", min_obstacles)
        object.__setattr__(self, "max_obstacles", max_obstacles)
        object.__setattr__(self, "require_path_blocker", blocker)


@dataclass(frozen=True)
class PyBulletCurriculumSettings:
    enabled: bool = False
    phases: tuple[PyBulletCurriculumPhaseSettings, ...] = ()

    def __post_init__(self) -> None:
        enabled = _bool_value("enabled", self.enabled)
        phases = tuple(self.phases)
        if enabled and not phases:
            raise ValueError("phases must not be empty")
        boundaries = tuple(phase.end_fraction for phase in phases)
        if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("end_fraction values must be strictly increasing")
        if phases and not math.isclose(boundaries[-1], 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("final curriculum end_fraction must equal 1.0")
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "phases", phases)

    def phase_for(self, progress: float) -> PyBulletCurriculumPhaseSettings:
        numeric = _unit_interval_float("progress", progress)
        if not self.enabled or not self.phases:
            raise ValueError("curriculum is not enabled")
        return next((phase for phase in self.phases if numeric < phase.end_fraction), self.phases[-1])
```

Add `_non_negative_int`, mapping-to-phase parsing, `vehicle_radius: float = 0.0`, and the two new fields on `TrainingSettings`. Export all three types from `swift.config`.

- [ ] **Step 4: Update the randomized YAML contract**

Add:

```yaml
pybullet_obstacle_randomization:
  vehicle_radius: 0.061
pybullet_reward:
  arrival_reward: 100.0
  approach_scale: 20.0
  collision_penalty: 100.0
  timeout_penalty: 20.0
  episode_time_penalty: 1.0
  heading_smoothness_penalty: 0.05
pybullet_curriculum:
  enabled: true
  phases:
    - name: goal_reaching
      end_fraction: 0.20
      min_obstacles: 0
      max_obstacles: 0
      require_path_blocker: false
    - name: single_obstacle
      end_fraction: 0.40
      min_obstacles: 1
      max_obstacles: 1
      require_path_blocker: false
    - name: single_blocker
      end_fraction: 0.70
      min_obstacles: 1
      max_obstacles: 1
      require_path_blocker: true
    - name: randomized_final
      end_fraction: 1.00
      min_obstacles: 1
      max_obstacles: 3
      require_path_blocker: true
run:
  total_timesteps: 150000
```

Update the existing strict-config test to assert these exact values.

- [ ] **Step 5: Run tests and commit**

```powershell
python -m pytest tests/config/test_training_settings.py -q
git add src/swift/config/settings.py src/swift/config/__init__.py configs/training_pybullet_randomized.yaml tests/config/test_training_settings.py
git commit -m "feat: configure pybullet curriculum rewards"
```

Expected: focused tests pass.

### Task 3: Make Randomized Geometry Vehicle-Aware

**Files:**
- Modify: `src/swift/envs/pybullet_obstacle_randomization.py:14-84`
- Test: `tests/envs/test_pybullet_obstacle_randomization.py`

- [ ] **Step 1: Strengthen the geometry tests**

Change the endpoint and blocker expectations to include the vehicle radius, and assert actionable exhaustion details:

```python
required_endpoint_distance = (
    obstacle.radius
    + randomization.vehicle_radius
    + environment.safety_margin
    + randomization.endpoint_clearance
)
assert math.dist(obstacle.position, environment.start) >= required_endpoint_distance
assert math.dist(obstacle.position, environment.goal) >= required_endpoint_distance

assert any(
    abs(obstacle.position[1])
    <= obstacle.radius + randomization.vehicle_radius + environment.safety_margin
    for obstacle in first
)
```

Add an exhaustion assertion:

```python
with pytest.raises(RuntimeError) as error:
    sample_pybullet_obstacles(randomization, environment, seed=9, phase_name="single_blocker")
message = str(error.value)
assert "seed=9" in message
assert "phase=single_blocker" in message
assert "required_endpoint_clearance=" in message
assert "obstacle_count=1..1" in message
```

- [ ] **Step 2: Run the test and verify the old point-geometry behavior fails**

```powershell
python -m pytest tests/envs/test_pybullet_obstacle_randomization.py -q
```

Expected: endpoint and message assertions fail.

- [ ] **Step 3: Update source geometry at the source of sampling**

Use the vehicle envelope in both checks:

```python
def _required_endpoint_distance(
    obstacle: ObstacleState,
    randomization: PyBulletObstacleRandomizationSettings,
    environment: SimpleAvoidanceSettings,
) -> float:
    return (
        obstacle.radius
        + randomization.vehicle_radius
        + environment.safety_margin
        + randomization.endpoint_clearance
    )


def _blocks_direct_path(
    obstacle: ObstacleState,
    randomization: PyBulletObstacleRandomizationSettings,
    environment: SimpleAvoidanceSettings,
) -> bool:
    start = environment.start
    goal = environment.goal
    segment = tuple(goal[index] - start[index] for index in range(3))
    segment_length_squared = sum(component * component for component in segment)
    if segment_length_squared <= 0.0:
        return False
    offset = tuple(obstacle.position[index] - start[index] for index in range(3))
    projection = sum(offset[index] * segment[index] for index in range(3)) / segment_length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = tuple(start[index] + projection * segment[index] for index in range(3))
    return math.dist(obstacle.position, nearest) <= (
        obstacle.radius + randomization.vehicle_radius + environment.safety_margin
    )
```

Add `phase_name: str = "final"` to `sample_pybullet_obstacles` and include all required context in the exhaustion error. Do not add runtime resampling or weaken determinism.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/envs/test_pybullet_obstacle_randomization.py tests/config/test_training_settings.py -q
git add src/swift/envs/pybullet_obstacle_randomization.py tests/envs/test_pybullet_obstacle_randomization.py
git commit -m "fix: account for pybullet vehicle clearance"
```

### Task 4: Apply Curriculum Phases And Reward Contract In The Environment

**Files:**
- Modify: `src/swift/envs/pybullet_velocity.py:23-239`
- Test: `tests/sim/test_pybullet_runtime_adapter.py:238-380,550-588`

- [ ] **Step 1: Add failing environment tests for phase snapshots**

Use the existing fake runtime and a four-phase curriculum:

```python
def test_pybullet_training_env_snapshots_curriculum_phase_on_reset(tmp_path: Path) -> None:
    reward = PyBulletRewardSettings(approach_scale=20.0, timeout_penalty=20.0)
    curriculum = PyBulletCurriculumSettings(
        enabled=True,
        phases=(
            PyBulletCurriculumPhaseSettings("goal_reaching", 0.20, 0, 0, False),
            PyBulletCurriculumPhaseSettings("single_obstacle", 0.40, 1, 1, False),
            PyBulletCurriculumPhaseSettings("single_blocker", 0.70, 1, 1, True),
            PyBulletCurriculumPhaseSettings("randomized_final", 1.00, 1, 3, True),
        ),
    )
    randomization = PyBulletObstacleRandomizationSettings(
        enabled=True,
        vehicle_radius=0.061,
    )
    env = PyBulletVelocityTrainingEnv(
        simulation_settings=make_settings(tmp_path),
        settings=SimpleAvoidanceSettings(
            start=(0.0, 0.0, 0.1125),
            goal=(0.5, 0.0, 0.1125),
            safety_margin=0.1,
        ),
        obstacle_randomization=randomization,
        reward_settings=reward,
        curriculum=curriculum,
        velocity_aviary_cls=lambda **_: FakeContactVelocityAviary(),
        drone_model=SimpleNamespace(CF2X="cf2x"),
        physics=SimpleNamespace(PYB="pyb"),
    )

    env.set_training_progress(0, 100)
    _, first = env.reset(seed=500000)
    env.set_training_progress(80, 100)
    _, final = env.reset(seed=500001)

    assert first["curriculum_phase"] == "goal_reaching"
    assert first["curriculum_progress"] == pytest.approx(0.0)
    assert first["obstacle_count"] == 0
    assert final["curriculum_phase"] == "randomized_final"
    assert final["curriculum_progress"] == pytest.approx(0.8)
    assert 1 <= final["obstacle_count"] <= 3
```

Also instantiate the environment without calling `set_training_progress` and assert reset selects `randomized_final` at progress `1.0`.

- [ ] **Step 2: Add failing reward and terminal-precedence tests**

Cover all terminal paths with configured values:

```python
assert progress_info["reward_breakdown"].approach == pytest.approx((0.05 / 2.0) * 20.0)
assert progress_info["reward_breakdown"].timeliness == pytest.approx(-1.0 / 500.0)

assert timeout_info["reward_breakdown"].timeliness == pytest.approx(-20.0 - 1.0 / max_steps)
assert collision_info["reward_breakdown"].obstacle == pytest.approx(-100.0)
assert collision_info["reward_breakdown"].arrive == pytest.approx(0.0)
assert collision_info["reached_goal"] is False
assert returned_reward == pytest.approx(collision_info["reward_breakdown"].total)
```

The collision-precedence fake must return an observation inside the goal radius and `collided=True` in the same step.

- [ ] **Step 3: Run focused tests and verify they fail**

```powershell
python -m pytest tests/sim/test_pybullet_runtime_adapter.py -q
```

Expected: failures for missing constructor settings, progress hook, phase metadata, and reward scaling.

- [ ] **Step 4: Implement phase selection and layout overrides**

Add constructor parameters for `reward_settings` and `curriculum`, defaulting to the backward-compatible config types. Keep progress at `1.0` unless the trainer changes it:

```python
def set_training_progress(self, completed_timesteps: int, total_timesteps: int) -> None:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if completed_timesteps < 0 or completed_timesteps > total_timesteps:
        raise ValueError("completed_timesteps must be between 0 and total_timesteps")
    self._training_progress = float(completed_timesteps) / float(total_timesteps)


def _snapshot_curriculum_phase(self) -> PyBulletCurriculumPhaseSettings | None:
    if not self._curriculum.enabled:
        return None
    return self._curriculum.phase_for(self._training_progress)
```

On reset, snapshot the phase. For a zero-obstacle phase install `()`. Otherwise use `dataclasses.replace` on the base randomization settings with phase obstacle counts and blocker requirement, then call the sampler with `phase_name=phase.name`. Do not mutate the base configuration.

```python
self._active_curriculum_phase = self._snapshot_curriculum_phase()
active_randomization = self._obstacle_randomization
if self._active_curriculum_phase is not None:
    phase = self._active_curriculum_phase
    if phase.max_obstacles == 0:
        self._active_obstacles = ()
    else:
        active_randomization = replace(
            self._obstacle_randomization,
            min_obstacles=phase.min_obstacles,
            max_obstacles=phase.max_obstacles,
            require_path_blocker=phase.require_path_blocker,
        )
        self._active_obstacles = sample_pybullet_obstacles(
            active_randomization,
            self.settings,
            seed=self._scenario_seed,
            phase_name=phase.name,
        )
elif active_randomization.enabled:
    self._active_obstacles = sample_pybullet_obstacles(
        active_randomization,
        self.settings,
        seed=self._scenario_seed,
        phase_name="final",
    )
self._runtime.set_swift_obstacles(self._active_obstacles)
```

- [ ] **Step 5: Implement the configured reward exactly once**

Calculate collision before successful arrival so terminal terms cannot both apply:

```python
within_goal = current_goal_distance <= self.settings.goal_radius
collided = bool(info.get("collided", False)) or _collided_from_safety(
    observation, current_safety_distance, self.settings.safety_margin
)
reached_goal = bool(within_goal and not collided)
terminated = bool(raw_terminated or reached_goal or collided)
timed_out = (
    not terminated
    and not raw_truncated
    and self._steps >= self.settings.max_steps
) or bool(info.get("timed_out", False))
truncated = bool(raw_truncated or timed_out)

reward_breakdown = RewardBreakdown(
    arrive=self._reward_settings.arrival_reward if reached_goal else 0.0,
    approach=normalized_approach * self._reward_settings.approach_scale,
    obstacle=-self._reward_settings.collision_penalty if collided else 0.0,
    smoothness=(
        -self._reward_settings.heading_smoothness_penalty
        * abs(float(drone_action.heading_delta))
    ),
    timeliness=(
        -self._reward_settings.episode_time_penalty / float(self.settings.max_steps)
        - (self._reward_settings.timeout_penalty if timed_out else 0.0)
    ),
)
```

Add phase/progress fields to reset, step, and terminal info only when curriculum is enabled. Always assert the total is finite before returning it.

- [ ] **Step 6: Verify old and new environment contracts and commit**

```powershell
python -m pytest tests/sim/test_pybullet_runtime_adapter.py tests/rl/test_torch_ppo.py -q
git add src/swift/envs/pybullet_velocity.py tests/sim/test_pybullet_runtime_adapter.py
git commit -m "feat: add pybullet obstacle curriculum environment"
```

### Task 5: Propagate PPO Progress And Aggregate Per-Phase Metrics

**Files:**
- Modify: `src/swift/rl/ppo.py:150-163`
- Modify: `src/swift/rl/__init__.py`
- Modify: `src/swift/rl/torch_ppo.py:137-345`
- Test: `tests/rl/test_torch_ppo.py:91-247`

- [ ] **Step 1: Add a curriculum-aware fake environment test**

Define a one-step fake that records progress and reports its phase:

```python
class CurriculumTrackingEnv(ClosingEnv):
    def __init__(self) -> None:
        super().__init__()
        self.progress_calls: list[tuple[int, int]] = []
        self.current_phase = "goal_reaching"

    def set_training_progress(self, completed_timesteps: int, total_timesteps: int) -> None:
        self.progress_calls.append((completed_timesteps, total_timesteps))
        self.current_phase = "goal_reaching" if completed_timesteps < total_timesteps // 2 else "randomized_final"

    def step(self, action):
        observation, reward, _, _, info = super().step(action)
        info.update(
            {
                "curriculum_phase": self.current_phase,
                "reached_goal": True,
                "collided": False,
                "timed_out": False,
            }
        )
        return observation, reward, True, False, info
```

Train for 8 steps and assert:

```python
assert env.progress_calls[0] == (0, 8)
assert env.progress_calls == sorted(env.progress_calls)
assert {item.phase for item in result.phase_metrics} == {"goal_reaching", "randomized_final"}
assert sum(item.episodes_completed for item in result.phase_metrics) == result.episodes_completed
```

Also retain an assertion that `SimpleAvoidanceEnv`, which has no progress hook or phase metadata, trains unchanged with `phase_metrics == ()`.

- [ ] **Step 2: Run the focused test and verify it fails**

```powershell
python -m pytest tests/rl/test_torch_ppo.py -q
```

Expected: missing progress calls and missing `phase_metrics`.

- [ ] **Step 3: Add immutable phase metrics to the public result**

```python
@dataclass(frozen=True)
class PPOPhaseMetrics:
    phase: str
    episodes_completed: int
    average_episode_return: float
    success_rate: float
    collision_rate: float
    timeout_rate: float


@dataclass(frozen=True)
class PPOTrainingResult:
    total_timesteps: int
    updates: int
    episodes_completed: int
    average_episode_return: float
    success_rate: float
    collision_rate: float
    timeout_rate: float
    final_policy_loss: float
    final_value_loss: float
    final_entropy: float
    checkpoint_path: str | None = None
    history_path: str | None = None
    phase_metrics: tuple[PPOPhaseMetrics, ...] = ()
```

Import `PPOPhaseMetrics` from `swift.rl.ppo` in `src/swift/rl/__init__.py` and add the exact string `"PPOPhaseMetrics"` to `__all__` next to `"PPOTrainingResult"`.

- [ ] **Step 4: Propagate progress only through an optional hook**

```python
def _set_training_progress(env: Any, completed: int, total: int) -> None:
    setter = getattr(env, "set_training_progress", None)
    if setter is not None:
        setter(completed, total)
```

Call it with `(0, total_timesteps)` before the first reset. Add the keyword-only parameter `starting_timestep: int` between `seed_offset` and `current_episode_return` in `_collect_rollout`, then pass and use it as follows:

```python
rollout = _collect_rollout(
    env=env,
    model=model,
    observation=observation,
    config=config,
    rollout_length=rollout_length,
    seed_offset=episodes_completed,
    starting_timestep=total_timesteps,
    current_episode_return=current_episode_return,
)

for local_step in range(rollout_length):
    observations.append(_observation_tensor(observation))
    action, raw_action, logprob, value = sample_action(
        model, observation, env.settings, config.network
    )
    next_observation, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    if done:
        completed = starting_timestep + local_step + 1
        _set_training_progress(env, completed, config.total_timesteps)
```

No phase changes occur in the middle of an episode because the environment snapshots only during reset.

- [ ] **Step 5: Aggregate terminal phase records without changing non-curriculum behavior**

Have `_collect_rollout` capture the completed return before resetting it, and return terminal records only when `curriculum_phase` exists:

```python
if done:
    episodes_completed += 1
    completed_return = current_episode_return
    episode_returns.append(completed_return)
    current_episode_return = 0.0
    success = _episode_success(info)
    collision = bool(info.get("collided", False))
    timeout = bool(info.get("timed_out", truncated))
    successes += int(success)
    collisions += int(collision)
    timeouts += int(timeout)
    if "curriculum_phase" in info:
        phase_episodes.append(
            {
                "phase": str(info["curriculum_phase"]),
                "return": completed_return,
                "success": success,
                "collision": collision,
                "timeout": timeout,
            }
        )
    completed = starting_timestep + local_step + 1
    _set_training_progress(env, completed, config.total_timesteps)
    next_observation, _ = env.reset(seed=config.seed + seed_offset + episodes_completed)
```

Merge records across updates and build sorted values with one helper:

```python
def _summarize_phase_metrics(records: Sequence[dict[str, Any]]) -> tuple[PPOPhaseMetrics, ...]:
    summaries = []
    for phase in sorted({str(record["phase"]) for record in records}):
        selected = [record for record in records if record["phase"] == phase]
        count = len(selected)
        summaries.append(
            PPOPhaseMetrics(
                phase=phase,
                episodes_completed=count,
                average_episode_return=_rate_sum([float(record["return"]) for record in selected]),
                success_rate=_rate(sum(bool(record["success"]) for record in selected), count),
                collision_rate=_rate(sum(bool(record["collision"]) for record in selected), count),
                timeout_rate=_rate(sum(bool(record["timeout"]) for record in selected), count),
            )
        )
    return tuple(summaries)
```

Include `[asdict(item) for item in phase_metrics]` in every update-history record and set the tuple on the checkpoint result.

- [ ] **Step 6: Verify serialization and commit**

```powershell
python -m pytest tests/rl/test_torch_ppo.py -q
git add src/swift/rl/ppo.py src/swift/rl/__init__.py src/swift/rl/torch_ppo.py tests/rl/test_torch_ppo.py
git commit -m "feat: report pybullet curriculum phase metrics"
```

### Task 6: Record Training Lineage And Final-Phase Evaluation Diagnostics

**Files:**
- Modify: `src/swift/experiments/pybullet_training_runner.py:45-219`
- Modify: `src/swift/experiments/pybullet_checkpoint_evaluator.py:56-197`
- Test: `tests/experiments/test_pybullet_training_runner.py`
- Test: `tests/experiments/test_pybullet_checkpoint_evaluator.py`

- [ ] **Step 1: Add failing training-runner lineage assertions**

Extend the randomized runner test:

```python
assert summary["runtime"]["obstacle_randomization"]["vehicle_radius"] == pytest.approx(0.061)
assert summary["runtime"]["reward"]["approach_scale"] == pytest.approx(20.0)
assert summary["runtime"]["curriculum"]["enabled"] is True
assert summary["runtime"]["curriculum"]["phases"][-1]["name"] == "randomized_final"
assert summary["training"]["phase_metrics"]
```

Capture the constructed environment and assert the runner passes the exact reward and curriculum objects from `TrainingSettings`.

- [ ] **Step 2: Add failing evaluator diagnostics and final-phase assertions**

Make the fake evaluation observations contain positions and goal distance, then assert:

```python
episode = summary["episodes"][0]
assert episode["curriculum_phase"] == "randomized_final"
assert episode["initial_goal_distance"] == pytest.approx(0.5)
assert episode["minimum_goal_distance"] == pytest.approx(0.3)
assert episode["final_goal_distance"] == pytest.approx(0.3)
assert episode["final_position"] == pytest.approx((0.2, 0.0, 0.1125))
assert summary["evaluation_scenarios"] == {"seed_start": 500000, "seed_end": 500000}
```

Add a fake whose reset reports `goal_reaching`; evaluation must raise `RuntimeError("evaluation must use final curriculum phase")`.

- [ ] **Step 3: Run focused tests and verify failures**

```powershell
python -m pytest tests/experiments/test_pybullet_training_runner.py tests/experiments/test_pybullet_checkpoint_evaluator.py -q
```

- [ ] **Step 4: Pass settings and serialize complete lineage**

Construct every training/evaluation environment with:

```python
reward_settings=config.training_settings.pybullet_reward,
curriculum=config.training_settings.pybullet_curriculum,
```

Add `reward` and `curriculum` alongside `obstacle_randomization` in runtime report metadata. They automatically participate in the existing config hash through `TrainingSettings`.

- [ ] **Step 5: Collect compact diagnostics during evaluation**

Capture reset info and update minimum goal distance in the loop:

```python
observation, reset_info = env.reset(seed=seed)
initial_goal_distance = float(observation[14])
minimum_goal_distance = initial_goal_distance
while not terminated and not truncated:
    action = deterministic_action(model, observation, env.settings, model.config)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    minimum_goal_distance = min(minimum_goal_distance, float(observation[14]))
```

When curriculum is enabled, compare `reset_info["curriculum_phase"]` with `training_settings.pybullet_curriculum.phases[-1].name`. Add final position and the three goal distances to the episode record. Add `evaluation_scenarios` to the summary.

- [ ] **Step 6: Verify and commit**

```powershell
python -m pytest tests/experiments/test_pybullet_training_runner.py tests/experiments/test_pybullet_checkpoint_evaluator.py tests/experiments/test_pybullet_multi_seed_training_runner.py tests/experiments/test_pybullet_multi_seed_checkpoint_evaluator.py -q
git add src/swift/experiments/pybullet_training_runner.py src/swift/experiments/pybullet_checkpoint_evaluator.py tests/experiments/test_pybullet_training_runner.py tests/experiments/test_pybullet_checkpoint_evaluator.py
git commit -m "feat: add pybullet curriculum evaluation evidence"
```

### Task 7: Add A Real-Runtime Geometry Gate

**Files:**
- Create: `src/swift/experiments/pybullet_geometry_validator.py`
- Modify: `src/swift/experiments/__init__.py`
- Create: `scripts/run_pybullet_geometry_check.py`
- Create: `tests/experiments/test_pybullet_geometry_validator.py`
- Modify: `tests/test_pybullet_randomized_training_scripts.py`

- [ ] **Step 1: Write a failing runner test**

Patch the environment with deterministic reset clearances and first-step outcomes. Assert the report is strict and complete:

```python
def test_pybullet_geometry_validation_reports_valid_final_phase_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swift.experiments.pybullet_geometry_validator as validator

    training_settings = TrainingSettings(
        environment=SimpleAvoidanceSettings(safety_margin=0.1),
        artifact=ExperimentArtifactConfig(
            root=tmp_path / "outputs",
            episode_logs=tmp_path / "episodes",
            experiment_reports=tmp_path / "reports",
            checkpoints=tmp_path / "checkpoints",
        ),
        pybullet_obstacle_randomization=PyBulletObstacleRandomizationSettings(enabled=True),
        pybullet_curriculum=PyBulletCurriculumSettings(
            enabled=True,
            phases=(
                PyBulletCurriculumPhaseSettings("randomized_final", 1.0, 1, 3, True),
            ),
        ),
    )
    simulation_settings = SimulationSettings(
        pybullet_root=tmp_path,
        pixi_executable=tmp_path / "pixi.exe",
        required_tasks=(),
        check_task="test",
        smoke_task="drone-demo",
        command_timeout_seconds=30,
    )

    class FakeGeometryEnv:
        settings = training_settings.environment

        def reset(self, seed=None):
            return (0.0,) * 15, {
                "scenario_seed": seed,
                "curriculum_phase": "randomized_final",
                "obstacle_count": 1,
                "minimum_safety_distance": 0.12,
            }

        def step(self, action):
            return (0.0,) * 15, 0.0, False, False, {"collided": False}

        def close(self):
            return None

    monkeypatch.setattr(validator, "PyBulletVelocityTrainingEnv", lambda **kwargs: FakeGeometryEnv())
    output_path = tmp_path / "geometry.json"
    report = run_pybullet_geometry_validation(
        PyBulletGeometryValidationConfig(
            training_settings=training_settings,
            simulation_settings=simulation_settings,
            seed=500000,
            scenarios=3,
            output=output_path,
        )
    )

    assert report["record_type"] == "pybullet_geometry_validation_report"
    assert report["scenarios"] == {"seed_start": 500000, "seed_end": 500002, "count": 3}
    assert report["metrics"]["invalid_initial_clearance_count"] == 0
    assert report["metrics"]["first_step_collision_count"] == 0
    assert report["readiness"]["geometry_valid"] is True
    assert all(item["curriculum_phase"] == "randomized_final" for item in report["layouts"])
```

Add a second case with clearance exactly equal to the safety margin because the environment collision rule uses `<= safety_margin`:

```python
    class BoundaryGeometryEnv(FakeGeometryEnv):
        def reset(self, seed=None):
            observation, info = super().reset(seed=seed)
            return observation, {**info, "minimum_safety_distance": 0.1}

    monkeypatch.setattr(
        validator,
        "PyBulletVelocityTrainingEnv",
        lambda **kwargs: BoundaryGeometryEnv(),
    )
    boundary = run_pybullet_geometry_validation(
        PyBulletGeometryValidationConfig(
            training_settings=training_settings,
            simulation_settings=simulation_settings,
            seed=500100,
            scenarios=1,
            output=tmp_path / "boundary.json",
        )
    )
    assert boundary["metrics"]["invalid_initial_clearance_count"] == 1
    assert boundary["readiness"]["geometry_valid"] is False
```

- [ ] **Step 2: Write a failing CLI contract test**

Add `import importlib.util`, `from types import SimpleNamespace`, and `import pytest` to the script test module.

Invoke the script in dry-run mode:

```python
def test_pybullet_geometry_check_script_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_geometry_check.py",
            "--seed",
            "500000",
            "--scenarios",
            "100",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "seed=500000" in result.stdout
    assert "scenarios=100" in result.stdout
```

Test exit policy without launching PyBullet:

```python
def test_pybullet_geometry_check_script_fails_on_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_pybullet_geometry_check",
        ROOT / "scripts" / "run_pybullet_geometry_check.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "geometry.json"

    monkeypatch.setattr(
        module,
        "_load_validator",
        lambda: (
            SimpleNamespace,
            lambda config: {
                "readiness": {"geometry_valid": False},
                "metrics": {
                    "invalid_initial_clearance_count": 1,
                    "first_step_collision_count": 0,
                },
                "artifacts": {"summary_json": str(config.output)},
            },
        ),
    )
    monkeypatch.setattr(module, "_load_runtime_unavailable_error", lambda: RuntimeError)

    exit_code = module.main(
        ["--scenarios", "1", "--output", str(output), "--fail-on-invalid"]
    )

    assert exit_code == 1
```

- [ ] **Step 3: Run tests and verify missing-module failures**

```powershell
python -m pytest tests/experiments/test_pybullet_geometry_validator.py tests/test_pybullet_randomized_training_scripts.py -q
```

- [ ] **Step 4: Implement the geometry validator**

The runner creates one environment, relies on default final-phase progress, and evaluates every seed:

```python
for scenario_seed in range(config.seed, config.seed + config.scenarios):
    _, reset_info = env.reset(seed=scenario_seed)
    initial_clearance = float(reset_info["minimum_safety_distance"])
    _, _, _, _, step_info = env.step(DroneAction(speed=0.0, heading_delta=0.0, climb_rate=0.0))
    layouts.append(
        {
            "seed": scenario_seed,
            "curriculum_phase": str(reset_info.get("curriculum_phase", "")),
            "obstacle_count": int(reset_info.get("obstacle_count", 0)),
            "initial_minimum_safety_distance": initial_clearance,
            "initial_clearance_valid": initial_clearance > env.settings.safety_margin,
            "first_step_collided": bool(step_info.get("collided", False)),
        }
    )
```

Validate finite clearances, close the environment in `finally`, write summary and manifest using the existing artifact helpers, and set `geometry_valid` only when both failure counts are zero.

- [ ] **Step 5: Implement the CLI and exports**

Required CLI flags:

```text
--training-config
--simulation-config
--seed (default 500000)
--scenarios (default 100)
--output
--dry-run
--fail-on-invalid
```

Return `2` for unavailable PyBullet, `1` for a completed invalid geometry report when requested, and `0` otherwise.

- [ ] **Step 6: Verify and commit**

```powershell
python -m pytest tests/experiments/test_pybullet_geometry_validator.py tests/test_pybullet_randomized_training_scripts.py -q
git add src/swift/experiments/pybullet_geometry_validator.py src/swift/experiments/__init__.py scripts/run_pybullet_geometry_check.py tests/experiments/test_pybullet_geometry_validator.py tests/test_pybullet_randomized_training_scripts.py
git commit -m "feat: gate randomized pybullet geometry"
```

### Task 8: Update CLI Defaults, Documentation, And Full Regression Coverage

**Files:**
- Modify: `scripts/run_pybullet_multi_seed_training.py`
- Modify: `scripts/run_pybullet_multi_seed_checkpoint_eval.py`
- Modify: `tests/test_pybullet_randomized_training_scripts.py`
- Modify: `README.md`
- Modify: `docs/implementation-roadmap.md`

- [ ] **Step 1: Write failing CLI default tests**

Assert training dry-run without `--total-timesteps` reports `150000` from YAML and the default output contains `3x150k`. Assert checkpoint evaluation still accepts arbitrary validation seeds but no longer presents the consumed `1000000` range as its documented/default final evidence.

```python
def test_pybullet_multi_seed_training_script_uses_yaml_timestep_default() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pybullet_multi_seed_training.py",
            "--training-config",
            "configs/training_pybullet_randomized.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "total_timesteps=150000" in result.stdout
    assert "pybullet_randomized_3x150k.json" in result.stdout
```

- [ ] **Step 2: Run script tests and verify old defaults fail**

```powershell
python -m pytest tests/test_pybullet_randomized_training_scripts.py -q
```

- [ ] **Step 3: Make YAML the single source of truth for timesteps**

Set `--total-timesteps` default to `None`; pass it through unchanged so `PyBulletMultiSeedTrainingConfig` uses `training_settings.run.total_timesteps`. Rename default artifacts to `3x150k`. Keep explicit overrides working for smoke and pilot runs.

- [ ] **Step 4: Document the exact staged workflow and seed boundaries**

Document these facts without claiming unrun success:

```text
Development validation: 500000..500099
Consumed holdout, never tune again: 1000000..1000099
Reserved final holdout: 2000000..2000099
Escalation: geometry -> 2,048 smoke -> 32,768 pilot -> 100,000 candidate -> 3 x 150,000
```

Include commands from Task 9 and state that simulator robustness is not real-flight safety evidence.

- [ ] **Step 5: Run the full repository verification**

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass, compilation exits `0`, and diff check prints nothing.

- [ ] **Step 6: Commit**

```powershell
git add scripts/run_pybullet_multi_seed_training.py scripts/run_pybullet_multi_seed_checkpoint_eval.py tests/test_pybullet_randomized_training_scripts.py README.md docs/implementation-roadmap.md
git commit -m "docs: define pybullet curriculum training workflow"
```

### Task 9: Execute The Staged Real PyBullet Training Gates

**Files:**
- Generated and ignored: `outputs/evaluation/pybullet_curriculum_geometry_validation.json`
- Generated and ignored: `outputs/training/pybullet_curriculum_smoke_1x2048.json`
- Generated and ignored: `outputs/training/pybullet_curriculum_pilot_1x32768.json`
- Generated and ignored: `outputs/evaluation/pybullet_curriculum_pilot_validation.json`
- Generated and ignored: `outputs/training/pybullet_curriculum_candidate_1x100k.json`
- Generated and ignored: `outputs/evaluation/pybullet_curriculum_candidate_validation.json`
- Generated and ignored only after gates pass: `outputs/training/pybullet_curriculum_3x150k.json`
- Generated and ignored only once: `outputs/evaluation/pybullet_curriculum_3x150k_final_holdout.json`
- Generated and ignored only once: `outputs/evaluation/pybullet_curriculum_3x150k_gate.json`

- [ ] **Step 1: Run the 100-layout real-runtime geometry gate**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_geometry_check.py --seed 500000 --scenarios 100 --output outputs\evaluation\pybullet_curriculum_geometry_validation.json --fail-on-invalid
```

Expected: exit `0`, zero invalid initial clearances, zero first-step collisions, and all layouts report `randomized_final`. Stop and diagnose any failure; do not train around invalid geometry.

- [ ] **Step 2: Run the 2,048-step finite-update smoke**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 2048 --output outputs\training\pybullet_curriculum_smoke_1x2048.json
```

Expected: one completed seed, finite metrics, checkpoint and JSONL artifacts present.

- [ ] **Step 3: Run and evaluate the 32,768-step short pilot**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 32768 --output outputs\training\pybullet_curriculum_pilot_1x32768.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_pilot_1x32768.json --episodes 20 --holdout-seed 500000 --output outputs\evaluation\pybullet_curriculum_pilot_validation.json
```

Enforce the gate:

```powershell
$train = Get-Content -Raw outputs\training\pybullet_curriculum_pilot_1x32768.json | ConvertFrom-Json
$eval = Get-Content -Raw outputs\evaluation\pybullet_curriculum_pilot_validation.json | ConvertFrom-Json
$phase = $train.seed_runs[0].training.phase_metrics | Where-Object phase -eq 'randomized_final'
if (-not $phase -or $phase.episodes_completed -lt 1) { throw 'pilot has no final-phase episodes' }
if ($eval.metrics.worst_success_rate -lt 0.20) { throw 'pilot success gate failed' }
```

Expected: at least one final-phase training episode and validation success at least `0.20`. If it fails, inspect compact episode diagnostics and form one new hypothesis before any retry.

- [ ] **Step 4: Run and evaluate the 100,000-step candidate pilot**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 --total-timesteps 100000 --output outputs\training\pybullet_curriculum_candidate_1x100k.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_candidate_1x100k.json --episodes 100 --holdout-seed 500000 --output outputs\evaluation\pybullet_curriculum_candidate_validation.json
```

Enforce all candidate metrics:

```powershell
$eval = Get-Content -Raw outputs\evaluation\pybullet_curriculum_candidate_validation.json | ConvertFrom-Json
if ($eval.metrics.worst_success_rate -lt 0.80) { throw 'candidate success gate failed' }
if ($eval.metrics.max_collision_rate -gt 0.05) { throw 'candidate collision gate failed' }
if ($eval.metrics.max_timeout_rate -gt 0.20) { throw 'candidate timeout gate failed' }
```

- [ ] **Step 5: Train the full candidate only after every earlier gate passes**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_training.py --seeds 8 9 10 --total-timesteps 150000 --output outputs\training\pybullet_curriculum_3x150k.json
```

Expected: all three child runs complete and aggregate timesteps equal `450000`.

- [ ] **Step 6: Consume the reserved final holdout exactly once**

```powershell
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_multi_seed_checkpoint_eval.py --input outputs\training\pybullet_curriculum_3x150k.json --episodes 100 --holdout-seed 2000000 --output outputs\evaluation\pybullet_curriculum_3x150k_final_holdout.json
D:\project\.venvs\swift-pybullet-pixi\Scripts\python.exe scripts\run_pybullet_robustness_gate.py --input outputs\evaluation\pybullet_curriculum_3x150k_final_holdout.json --output outputs\evaluation\pybullet_curriculum_3x150k_gate.json --fail-on-reject
```

Expected for a robustness claim: exit `0`, worst success at least `0.95`, maximum collision `0.0`, maximum timeout at most `0.05`, and worst average minimum safety distance at least `0.10 m`. Report failure honestly and do not tune against this final holdout.

- [ ] **Step 7: Finish with a clean verification record**

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
git status --short
```

Expected: tests and checks pass; only ignored generated artifacts remain outside Git status.
