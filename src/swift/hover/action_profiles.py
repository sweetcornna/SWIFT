"""SWIFT-owned stabilized-hover action contract.

The numerical controller intentionally mirrors the validated external substrate's
``scripts/action_profiles.py`` contract. The environment class itself remains
external and is adapted at runtime; no simulator source is vendored into SWIFT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PROFILE_CT_ATT_YAWRATE_V1 = "ct_att_yawrate_v1"
MIXER_CF2X = np.array(((-.5, -.5, -1.), (-.5, .5, 1.), (.5, .5, -1.), (.5, -.5, 1.)), dtype=np.float64)
KP = np.array((70000., 70000., 60000.))
KI = np.array((0., 0., 500.))
KD = np.array((20000., 20000., 12000.))
INTEGRAL_CLIP = np.array((1., 1., 1500.))
PWM_MIN, PWM_MAX = 20000., 65535.
PWM2RPM_SCALE, PWM2RPM_CONST = .2685, 4070.3
TORQUE_CLIP = 3200.


def wrap_angle(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.arctan2(np.sin(value), np.cos(value))


def _quaternion_to_matrix_and_rpy(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the scipy-compatible rotation matrix and extrinsic XYZ angles."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    matrix = np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-12:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return matrix, np.array((roll, pitch, yaw), dtype=np.float64)


def _euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def map_action(action: Any, mass: float, gravity: float) -> tuple[float, np.ndarray, np.ndarray]:
    command = np.asarray(action, dtype=np.float32)
    if command.shape != (1, 4) or not np.all(np.isfinite(command)):
        raise ValueError("ct_att_yawrate_v1 action must be finite float-compatible shape (1,4)")
    command = np.clip(command, -1., 1.).astype(np.float32)
    thrust = float(mass * gravity * (1. + .5 * float(command[0, 0])))
    target_rpy = np.array((.25 * command[0, 1], .25 * command[0, 2], 0.), dtype=np.float64)
    target_rates = np.array((0., 0., 2. * command[0, 3]), dtype=np.float64)
    return thrust, target_rpy, target_rates


@dataclass(frozen=True)
class ControllerResult:
    rpm: np.ndarray
    pwm: np.ndarray
    saturated: np.ndarray
    torque: np.ndarray


class TransactionalDSLPIDAttitude:
    """CF2X DSLPID-equivalent loop that commits state only after finite output."""

    def __init__(self, *, kf: float, mass: float, gravity: float, timestep: float):
        values = np.asarray((kf, mass, gravity, timestep), dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("kf, mass, gravity, and timestep must be finite and positive")
        self.kf, self.mass, self.gravity, self.timestep = values.tolist()
        self.reset(np.zeros(3, dtype=np.float64))

    def reset(self, realized_rpy: Any) -> None:
        rpy = np.asarray(realized_rpy, dtype=np.float64)
        if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
            raise ValueError("realized reset RPY must be finite shape (3,)")
        self.control_counter = 0
        self.last_rpy = rpy.copy()
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.last_result: ControllerResult | None = None

    def compute(self, action: Any, cur_quat: Any) -> ControllerResult:
        force, target_euler, target_rates = map_action(action, self.mass, self.gravity)
        cur_quat = np.asarray(cur_quat, dtype=np.float64)
        if cur_quat.shape != (4,) or not np.all(np.isfinite(cur_quat)):
            raise ValueError("current quaternion must be finite shape (4,)")
        norm = float(np.linalg.norm(cur_quat))
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise ValueError("current quaternion must have nonzero finite norm")
        cur_rotation, cur_rpy = _quaternion_to_matrix_and_rpy(cur_quat)
        target_euler[2] = cur_rpy[2]
        target_rotation = _euler_xyz_to_matrix(target_euler)
        skew = target_rotation.T @ cur_rotation - cur_rotation.T @ target_rotation
        rot_e = np.array((skew[2, 1], skew[0, 2], skew[1, 0]))
        measured_rates = wrap_angle(cur_rpy - self.last_rpy) / self.timestep
        rates_e = target_rates - measured_rates
        integral = np.clip(self.integral_rpy_e - rot_e * self.timestep, -INTEGRAL_CLIP, INTEGRAL_CLIP)
        torque = np.clip(-KP * rot_e + KD * rates_e + KI * integral, -TORQUE_CLIP, TORQUE_CLIP)
        collective_pwm = (np.sqrt(force / (4. * self.kf)) - PWM2RPM_CONST) / PWM2RPM_SCALE
        raw_pwm = collective_pwm + MIXER_CF2X @ torque
        pwm = np.clip(raw_pwm, PWM_MIN, PWM_MAX)
        rpm = PWM2RPM_SCALE * pwm + PWM2RPM_CONST
        if not all(np.all(np.isfinite(value)) for value in (cur_rpy, integral, torque, pwm, rpm)):
            raise FloatingPointError("attitude controller produced non-finite output")
        result = ControllerResult(rpm.astype(np.float64), pwm.astype(np.float64), raw_pwm != pwm, torque.astype(np.float64))
        self.control_counter += 1
        self.last_rpy = cur_rpy.copy()
        self.last_rpy_e = rot_e.copy()
        self.integral_rpy_e = integral
        self.last_result = result
        return result


def action_profile_metadata() -> dict[str, Any]:
    return {
        "name": PROFILE_CT_ATT_YAWRATE_V1,
        "schema_version": 1,
        "policy_action": ["collective", "desired_roll", "desired_pitch", "desired_euler_yaw_rate"],
        "shape": [1, 4], "dtype": "float32", "bounds": [-1.0, 1.0],
        "mapping": {"total_thrust_N": "m*g*(1+0.5*a0)", "roll_rad": "0.25*a1", "pitch_rad": "0.25*a2", "yaw_rate_rad_s": "2*a3"},
        "controller": {"family": "root_owned_transactional_dslpid_attitude_equivalent", "mixer": "CF2X", "kp": KP.tolist(), "ki": KI.tolist(), "kd": KD.tolist(), "torque_clip": [-3200., 3200.], "integral_clips": INTEGRAL_CLIP.tolist(), "pwm_clip": [PWM_MIN, PWM_MAX], "rpm_formula": "0.2685*PWM+4070.3", "yaw_delta": "atan2(sin(delta),cos(delta))", "evaluation_frequency_hz": 30, "physics_hold_substeps": 8, "commit": "only_after_finite_output"},
        "reset": "after_all_physical_randomization_clear_counters_integrals_history_diagnostics_and_seed_last_rpy_from_realized_state",
        "observation": {"shape": [1, 72], "physical": [0, 12], "high_level_action_history": [12, 72], "generated_rpm_excluded": True},
    }
