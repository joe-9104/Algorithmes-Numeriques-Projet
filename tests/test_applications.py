import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure imports resolve to project packages even when discovery starts in tests/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.flow import FlowProblem
from models.temperatures import CoolingProblem


class TestCoolingProblem(unittest.TestCase):
    def test_temperature_matches_known_data_points(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        temp = np.array([90.0, 85.0, 72.0, 63.0])
        problem = CoolingProblem(t, temp)

        np.testing.assert_allclose(problem.temperature(t), temp)

    def test_heat_loss_rate_uses_h_and_ambient(self):
        t = np.array([0.0, 2.0, 4.0])
        temp = np.array([40.0, 40.0, 40.0])
        problem = CoolingProblem(t, temp, T_ambient=20.0, h_coeff=10.0)

        self.assertAlmostEqual(problem.heat_loss_rate(1.0), 200.0, places=9)

    def test_total_heat_loss_for_constant_temperature(self):
        t = np.array([0.0, 5.0, 10.0])
        temp = np.array([40.0, 40.0, 40.0])
        problem = CoolingProblem(t, temp, T_ambient=20.0, h_coeff=50.0)

        # Q = integral_0^10 50 * (40 - 20) dt = 10000
        self.assertAlmostEqual(problem.total_heat_loss(method="simpson", n=100), 10000.0, places=6)

    def test_estimate_k_recovers_synthetic_parameter(self):
        t = np.linspace(0.0, 10.0, 11)
        k_true = 0.12
        T_ambient = 20.0
        T0 = 90.0
        temp = T_ambient + (T0 - T_ambient) * np.exp(-k_true * t)

        problem = CoolingProblem(t, temp, T_ambient=T_ambient, h_coeff=50.0)
        k_est = problem.estimate_k(k_min=0.01, k_max=0.5, tol=1e-5)

        self.assertAlmostEqual(k_est, k_true, delta=5e-3)


class TestFlowProblem(unittest.TestCase):
    def test_velocity_matches_known_data_points(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 2.0, 4.0, 6.0])
        problem = FlowProblem(x, v)

        np.testing.assert_allclose(problem.velocity(x), v)

    def test_local_flow_rate_uses_velocity_and_width(self):
        x = np.array([0.0, 1.0, 2.0])
        v = np.array([1.0, 1.0, 1.0])
        problem = FlowProblem(x, v, width_func=lambda s: 2.0)

        self.assertAlmostEqual(problem.local_flow_rate(1.5), 2.0, places=9)

    def test_total_flow_rate_for_constant_fields(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([2.0, 2.0, 2.0, 2.0])
        problem = FlowProblem(x, v, width_func=lambda s: 1.0)

        # Q = integral_0^3 2 * 1 dx = 6
        self.assertAlmostEqual(problem.total_flow_rate(method="simpson", n=100), 6.0, places=6)

    def test_acceleration_matches_linear_profile(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        v = 3.0 * x + 2.0
        problem = FlowProblem(x, v)

        # v = 3x+2, dv/dx = 3, a = v * dv/dx = 3 * (3x + 2)
        self.assertAlmostEqual(problem.acceleration(1.0), 15.0, delta=1e-3)

    def test_work_matches_linear_profile(self):
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        v = 3.0 * x + 2.0
        problem = FlowProblem(x, v)

        # W = integral_0^2 m * 3 * (3x+2) dx for m=2 gives 60
        self.assertAlmostEqual(problem.work(mass=2.0), 60.0, delta=1e-2)


if __name__ == "__main__":
    unittest.main()
