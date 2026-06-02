from __future__ import annotations

from dataclasses import dataclass


class UnsupportedOperationError(RuntimeError):
    """Raised when a bootstrap contract is called as a runtime implementation."""


@dataclass(frozen=True)
class BootstrapDroneEnv:
    observation_size: int
    action_size: int

    @property
    def observation_shape(self) -> tuple[int]:
        return (self.observation_size,)

    @property
    def action_shape(self) -> tuple[int]:
        return (self.action_size,)

    def reset(self) -> None:
        raise UnsupportedOperationError("Full RL environment runtime is outside bootstrap scope")

    def step(self, action: object) -> None:
        raise UnsupportedOperationError("Full RL environment runtime is outside bootstrap scope")
