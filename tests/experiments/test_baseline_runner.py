import pytest

from swift.experiments import BaselineRunConfig, run_baseline_episodes
from swift.experiments import baseline_runner
from swift.rl import MLPBaselinePolicyConfig


class _FakeEnv:
    observation_shape = (15,)
    action_shape = (3,)

    _outcomes = (
        ("reached_goal", (1.0, 0.0, 0.0), (5.0, 0.0, 0.0), 0.5, True, False),
        ("collided", (0.5, 0.0, 0.0), (0.3, 0.0, 0.0), 0.5, True, False),
        ("timeout", (0.1, 0.0, 0.0), (3.0, 0.0, 0.0), 0.5, False, True),
    )

    def __init__(self, settings: object) -> None:
        self.settings = settings
        self.episode_index = -1

    def reset(self) -> tuple[tuple[float, ...], dict[str, object]]:
        self.episode_index += 1
        return _observation(position=(0.0, 0.0, 0.0)), {}

    def step(self, action: object) -> tuple[tuple[float, ...], float, bool, bool, dict[str, object]]:
        label, position, obstacle, radius, terminated, truncated = self._outcomes[self.episode_index]
        info = {
            "reached_goal": label == "reached_goal",
            "collided": label == "collided",
        }
        return _observation(position=position, obstacle_relative=obstacle, obstacle_radius=radius), 0.0, terminated, truncated, info


def _observation(
    *,
    position: tuple[float, float, float],
    obstacle_relative: tuple[float, float, float] = (10.0, 0.0, 0.0),
    obstacle_radius: float = 0.5,
) -> tuple[float, ...]:
    return (
        *position,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        *obstacle_relative,
        obstacle_radius,
        1.0,
    )


def test_run_baseline_episodes_reports_aggregate_fields(monkeypatch):
    monkeypatch.setattr(baseline_runner, "SimpleAvoidanceEnv", _FakeEnv)
    config = BaselineRunConfig(
        episodes=3,
        env_settings=object(),
        policy_config=MLPBaselinePolicyConfig(),
    )

    result = run_baseline_episodes(config)

    assert result["variant"] == "ppo_mlp_contract_baseline"
    assert result["episodes"] == 3
    assert result["success_rate"] == pytest.approx(1 / 3)
    assert result["collision_rate"] == pytest.approx(1 / 3)
    assert result["timeout_rate"] == pytest.approx(1 / 3)
    assert result["average_path_length"] > 0.0
    assert result["average_path_smoothness"] == pytest.approx(0.0)
    assert result["minimum_safety_distance"] == pytest.approx(-0.2)
    assert len(result["episodes_detail"]) == 3
    assert result["episodes_detail"][0]["success"] is True
    assert result["episodes_detail"][1]["collided"] is True
    assert result["episodes_detail"][2]["timed_out"] is True


def test_run_baseline_episodes_rejects_non_positive_episode_count():
    config = BaselineRunConfig(episodes=0, env_settings=object())

    with pytest.raises(ValueError, match="episodes"):
        run_baseline_episodes(config)
