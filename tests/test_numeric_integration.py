import math
import unittest

from core.numeric_integration import AdaptiveIntegration, GaussQuadrature, NewtonCotes


class TestNewtonCotes(unittest.TestCase):
    def test_rectangle_zero_interval_returns_zero(self):
        result = NewtonCotes.rectangle(lambda x: x**2, 2.0, 2.0, n=10)
        self.assertEqual(result, 0.0)

    def test_trapezoidal_exact_for_linear_function(self):
        # Integral of 3x + 2 over [0, 4] is 32.
        result = NewtonCotes.trapezoidal(lambda x: 3.0 * x + 2.0, 0.0, 4.0, n=8)
        self.assertAlmostEqual(result, 32.0, places=12)

    def test_simpson_exact_for_cubic_function(self):
        # Integral of x^3 over [0, 2] is 4.
        result = NewtonCotes.simpson(lambda x: x**3, 0.0, 2.0, n=10)
        self.assertAlmostEqual(result, 4.0, places=12)

    def test_simpson_38_exact_for_cubic_function(self):
        # Integral of x^3 over [0, 2] is 4.
        result = NewtonCotes.simpson_38(lambda x: x**3, 0.0, 2.0, n=6)
        self.assertAlmostEqual(result, 4.0, places=12)

    def test_simpson_requires_even_n(self):
        with self.assertRaises(ValueError):
            NewtonCotes.simpson(lambda x: x, 0.0, 1.0, n=3)

    def test_simpson_38_requires_multiple_of_three(self):
        with self.assertRaises(ValueError):
            NewtonCotes.simpson_38(lambda x: x, 0.0, 1.0, n=4)

    def test_methods_reject_bool_n(self):
        for method in (
            NewtonCotes.rectangle,
            NewtonCotes.trapezoidal,
            NewtonCotes.simpson,
            NewtonCotes.simpson_38,
        ):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method(lambda x: x, 0.0, 1.0, n=True)


class TestAdaptiveIntegration(unittest.TestCase):
    def test_constructor_validates_tol_and_max_depth(self):
        with self.assertRaises(ValueError):
            AdaptiveIntegration(tol=0.0, max_depth=10)

        with self.assertRaises(ValueError):
            AdaptiveIntegration(tol=1e-6, max_depth=0)

    def test_adaptive_simpson_validates_tol_override(self):
        integrator = AdaptiveIntegration(tol=1e-6, max_depth=20)
        with self.assertRaises(ValueError):
            integrator.adaptive_simpson(lambda x: x**2, 0.0, 1.0, tol=0.0)

    def test_adaptive_simpson_is_accurate_on_sine(self):
        integrator = AdaptiveIntegration(tol=1e-10, max_depth=30)
        result = integrator.adaptive_simpson(math.sin, 0.0, math.pi)
        self.assertAlmostEqual(result, 2.0, places=8)

    def test_adaptive_simpson_zero_interval_returns_zero(self):
        integrator = AdaptiveIntegration()
        result = integrator.adaptive_simpson(lambda x: math.exp(x), 1.5, 1.5)
        self.assertEqual(result, 0.0)


class TestGaussQuadrature(unittest.TestCase):
    def test_gauss_legendre_2_exact_for_cubic(self):
        # 2-point Gauss-Legendre is exact up to degree 3.
        result = GaussQuadrature.gauss_legendre_2(lambda x: x**3 - 2.0 * x + 1.0, -1.0, 2.0)
        expected = 3.75
        self.assertAlmostEqual(result, expected, places=12)

    def test_gauss_legendre_3_exact_for_quintic(self):
        # 3-point Gauss-Legendre is exact up to degree 5.
        result = GaussQuadrature.gauss_legendre_3(lambda x: x**5 - x**4 + 2.0 * x, -1.0, 1.0)
        expected = -2.0 / 5.0
        self.assertAlmostEqual(result, expected, places=12)

    def test_gauss_legendre_3_good_accuracy_on_exp(self):
        result = GaussQuadrature.gauss_legendre_3(math.exp, 0.0, 1.0)
        expected = math.e - 1.0
        self.assertAlmostEqual(result, expected, delta=1e-6)


if __name__ == "__main__":
    unittest.main()

