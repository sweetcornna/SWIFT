import numpy as np
import pytest

from swift.hover.action_profiles import TransactionalDSLPIDAttitude, map_action, wrap_angle


def test_map_action_enforces_contract_and_physical_mapping():
    thrust, rpy, rates = map_action([[0.5, 1.0, -1.0, 0.25]], mass=0.027, gravity=9.81)
    assert thrust == pytest.approx(0.027 * 9.81 * 1.25)
    np.testing.assert_allclose(rpy, [0.25, -0.25, 0.0])
    np.testing.assert_allclose(rates, [0.0, 0.0, 0.5])
    with pytest.raises(ValueError):
        map_action([0.0] * 4, mass=0.027, gravity=9.81)
    with pytest.raises(ValueError):
        map_action([[np.nan, 0.0, 0.0, 0.0]], mass=0.027, gravity=9.81)


def test_controller_is_transactional_on_invalid_input_and_finite_on_identity():
    controller = TransactionalDSLPIDAttitude(kf=3.16e-10, mass=0.027, gravity=9.81, timestep=1 / 30)
    result = controller.compute(np.zeros((1, 4), dtype=np.float32), [0.0, 0.0, 0.0, 1.0])
    assert controller.control_counter == 1
    assert result.rpm.shape == (4,)
    assert np.all(np.isfinite(result.rpm))
    state = (controller.control_counter, controller.last_rpy.copy(), controller.integral_rpy_e.copy())
    with pytest.raises(ValueError):
        controller.compute(np.zeros((1, 4), dtype=np.float32), [0.0, 0.0, 0.0, 0.0])
    assert controller.control_counter == state[0]
    np.testing.assert_array_equal(controller.last_rpy, state[1])
    np.testing.assert_array_equal(controller.integral_rpy_e, state[2])


def test_wrap_angle_handles_branch_cut():
    np.testing.assert_allclose(wrap_angle([2 * np.pi + 0.1, -2 * np.pi - 0.1]), [0.1, -0.1])
