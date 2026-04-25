import unittest
import sys
from pathlib import Path

import numpy as np

# Ensure imports resolve to project packages even when discovery starts in tests/.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.polynomial_interpolation import PolynomialInterpolation


class TestPolynomialInterpolation(unittest.TestCase):
    def test_init_raises_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            PolynomialInterpolation([0, 1], [1])

    def test_init_raises_on_duplicate_x_points(self):
        with self.assertRaises(ValueError):
            PolynomialInterpolation([0, 1, 1], [1, 2, 3])

    def test_lagrange_matches_nodes_exactly(self):
        x = np.array([-1.0, 0.0, 2.0, 4.0])
        y = x**3 - 2 * x + 1
        interp = PolynomialInterpolation(x, y)

        np.testing.assert_allclose(interp.lagrange(x), y)

    def test_newton_matches_nodes_exactly(self):
        x = np.array([-2.0, -0.5, 1.0, 3.0])
        y = 2 * x**2 + 3 * x - 5
        interp = PolynomialInterpolation(x, y)

        np.testing.assert_allclose(interp.newton_eval(x), y)

    def test_lagrange_and_newton_are_consistent_on_grid(self):
        x = np.array([0.0, 1.0, 2.0, 5.0])
        y = np.array([1.0, 3.0, 2.0, 8.0])
        x_eval = np.linspace(-1.0, 6.0, 25)
        interp = PolynomialInterpolation(x, y)

        np.testing.assert_allclose(interp.lagrange(x_eval), interp.newton_eval(x_eval), rtol=1e-10, atol=1e-12)

    def test_scalar_input_returns_scalar_value(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 2.0])
        interp = PolynomialInterpolation(x, y)

        lagrange_value = interp.lagrange(1.5)
        newton_value = interp.newton_eval(1.5)

        self.assertTrue(np.isscalar(lagrange_value))
        self.assertTrue(np.isscalar(newton_value))
        self.assertAlmostEqual(lagrange_value, newton_value, places=12)

    def test_evaluate_method_is_case_and_space_insensitive(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 2.0])
        x_eval = np.array([0.25, 1.5])
        interp = PolynomialInterpolation(x, y)

        np.testing.assert_allclose(interp.evaluate(x_eval, method=" newton "), interp.newton_eval(x_eval))
        np.testing.assert_allclose(interp.evaluate(x_eval, method="LAGRANGE"), interp.lagrange(x_eval))

    def test_evaluate_raises_on_unknown_method(self):
        interp = PolynomialInterpolation([0, 1], [1, 2])

        with self.assertRaises(ValueError):
            interp.evaluate([0.5], method="splines")


if __name__ == "__main__":
    unittest.main()


