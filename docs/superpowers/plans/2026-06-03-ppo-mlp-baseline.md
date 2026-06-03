# PPO MLP Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable Stage 1 baseline: a deterministic SWIFT-native 3D avoidance environment, a no-Torch MLP-shaped policy contract, an episode runner, and a JSON CLI smoke.

**Architecture:** Keep `D:\project\pybullet` as an external command-level substrate while SWIFT owns the first environment and experiment contracts. The environment exposes a Gymnasium-style API without adding Gymnasium as a dependency: `reset() -> (observation, info)` and `step(action) -> (observation, reward, terminated, truncated, info)`.

**Tech Stack:** Python 3.11+, standard library, existing SWIFT dataclasses and metrics, pytest. No Torch, Gymnasium, NumPy, or PyBullet import in this slice.

---

## File Structure

- `src/swift/envs/simple_avoidance.py`: deterministic 3D environment with 15-value observations, action coercion, rewards, termination, truncation, and episode metrics.
- `src/swift/envs/__init__.py`: exports `SimpleAvoidanceEnv` and `SimpleAvoidanceSettings` while preserving bootstrap exports.
- `tests/envs/test_simple_avoidance.py`: reset, step, goal, collision, timeout, reward, and dynamic obstacle coverage.
- `src/swift/rl/mlp_baseline.py`: deterministic policy contract shaped like the future MLP actor.
- `src/swift/rl/__init__.py`: exports baseline policy types.
- `tests/rl/test_mlp_baseline.py`: policy bounds, obstacle steering, and observation validation.
- `src/swift/experiments/baseline_runner.py`: repeated episode runner and aggregate metrics.
- `src/swift/experiments/__init__.py`: exports `BaselineRunConfig` and `run_baseline_episodes`.
- `tests/experiments/test_baseline_runner.py`: aggregate metric and validation tests.
- `scripts/run_baseline_demo.py`: CLI smoke that writes strict JSON metrics.
- `tests/test_baseline_script.py`: subprocess CLI test.
- `README.md`, `docs/implementation-roadmap.md`, `docs/architecture.md`, `docs/risk-register.md`: Stage 1 command and boundary documentation.

---

## Task 1: Simple Avoidance Environment

**Files:**
- Create: `src/swift/envs/simple_avoidance.py`
- Modify: `src/swift/envs/__init__.py`
- Modify: `tests/contracts/test_bootstrap_contracts.py`
- Test: `tests/envs/test_simple_avoidance.py`

- [x] **Step 1: Write failing tests**

Cover these exact behaviors:

```python
observation, info = env.reset(seed=123, options={"unused": True})
assert env.observation_shape == (15,)
assert env.action_shape == (3,)
assert "episode_metrics" not in info

observation, reward, terminated, truncated, info = env.step((1.0, 0.0, 0.0))
assert reward == info["reward_breakdown"].total
```

Run:

```powershell
python -m pytest tests\envs\test_simple_avoidance.py tests\contracts\test_bootstrap_contracts.py
```

Expected before implementation: missing `SimpleAvoidanceEnv` import.

- [x] **Step 2: Implement environment contract**

Implement:

```python
class SimpleAvoidanceEnv:
    @property
    def observation_shape(self) -> tuple[int]: return (15,)

    @property
    def action_shape(self) -> tuple[int]: return (3,)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[tuple[float, ...], dict[str, Any]]: ...

    def step(self, action: DroneAction | Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]: ...
```

Observation order is fixed:

```text
position3, velocity3, yaw1, relative_goal3, nearest_obstacle_relative3, nearest_obstacle_radius1, goal_distance1
```

- [x] **Step 3: Verify environment slice**

Run:

```powershell
python -m pytest tests\envs\test_simple_avoidance.py tests\contracts\test_bootstrap_contracts.py tests\core\test_metrics.py
```

Expected: selected tests pass.

---

## Task 2: MLP Baseline Policy and Runner

**Files:**
- Create: `src/swift/rl/mlp_baseline.py`
- Modify: `src/swift/rl/__init__.py`
- Create: `src/swift/experiments/baseline_runner.py`
- Modify: `src/swift/experiments/__init__.py`
- Test: `tests/rl/test_mlp_baseline.py`
- Test: `tests/experiments/test_baseline_runner.py`

- [x] **Step 1: Write failing policy and runner tests**

Cover:

```python
policy = MLPBaselinePolicy(MLPBaselinePolicyConfig(max_speed=2.0, max_heading_delta=0.3))
action = policy.act((0.0,) * 15)
assert 0.0 <= action.speed <= 2.0
assert -0.3 <= action.heading_delta <= 0.3

result = run_baseline_episodes(BaselineRunConfig(episodes=3))
assert result["variant"] == "ppo_mlp_contract_baseline"
assert result["episodes"] == 3
```

Run:

```powershell
python -m pytest tests\rl\test_mlp_baseline.py tests\experiments\test_baseline_runner.py
```

Expected before implementation: missing `MLPBaselinePolicy` and `BaselineRunConfig` imports.

- [x] **Step 2: Implement deterministic policy contract**

Implement:

```python
@dataclass(frozen=True)
class MLPBaselinePolicyConfig:
    max_speed: float = 1.0
    max_heading_delta: float = 0.5
    max_climb_rate: float = 0.5
    obstacle_avoidance_distance: float = 2.0
    avoidance_heading_delta: float = 0.4

class MLPBaselinePolicy:
    def act(self, observation: tuple[float, ...]) -> DroneAction: ...
```

- [x] **Step 3: Implement aggregate runner**

Return strict aggregate fields:

```python
{
    "variant": "ppo_mlp_contract_baseline",
    "episodes": episodes,
    "success_rate": ...,
    "collision_rate": ...,
    "timeout_rate": ...,
    "average_path_length": ...,
    "average_path_smoothness": ...,
    "minimum_safety_distance": ...,
    "episodes_detail": [...],
}
```

- [x] **Step 4: Verify policy and runner slice**

Run:

```powershell
python -m pytest tests\rl\test_mlp_baseline.py tests\experiments\test_baseline_runner.py
```

Expected: selected tests pass.

---

## Task 3: CLI and Documentation

**Files:**
- Create: `scripts/run_baseline_demo.py`
- Create: `tests/test_baseline_script.py`
- Modify: `README.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/architecture.md`
- Modify: `docs/risk-register.md`

- [x] **Step 1: Write failing CLI test**

The subprocess test must run:

```powershell
python scripts\run_baseline_demo.py --episodes 3 --output <temp-file>
```

and assert:

```python
assert metrics["episodes"] == 3
assert "success_rate" in metrics
assert "Infinity" not in raw_metrics
assert "NaN" not in raw_metrics
```

- [x] **Step 2: Implement CLI**

Use the existing script pattern:

```python
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

Write JSON with:

```python
json.dumps(metrics, allow_nan=False, indent=2, sort_keys=True)
```

- [x] **Step 3: Document Stage 1 boundaries**

Document that:

- Stage 1 baseline is repo-native first.
- `D:\project\pybullet` remains command-level smoke until the environment adapter stabilizes.
- Current adapter validates substrate and Pixi commands; future adapter translates SWIFT reset/step contracts.

---

## Final Verification

- [x] `python -m pytest`
- [x] `python scripts\swift_healthcheck.py`
- [x] `python scripts\run_pybullet_smoke.py --check-only`
- [x] `python scripts\run_baseline_demo.py --episodes 3 --output outputs\baseline\integration_metrics.json`

Expected: all commands exit 0.

## Self-Review

- Spec coverage: implements Stage 1's first runnable baseline loop, metrics output, repo-native environment interface, and PyBullet boundary preservation.
- Deferred deliberately: trainable Torch PPO weights, Gymnasium spaces, and live PyBullet runtime stepping. Those belong in the next Stage 1 slice after this contract is stable.
- Type consistency: observation length is consistently 15; policy consumes that same layout; runner expects Gymnasium-style reset/step returns.
