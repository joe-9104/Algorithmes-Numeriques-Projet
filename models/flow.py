from __future__ import annotations

import numpy as np

from core.numeric_integration import AdaptiveIntegration, GaussQuadrature, NewtonCotes
from core.polynomial_interpolation import PolynomialInterpolation


class FlowProblem:
	"""Modelise le probleme d'ecoulement 1D dans un canal."""

	def __init__(self, x_data, v_data, width_func=None):
		"""Initialise les donnees experimentales et la fonction de largeur."""

		self.x_data = np.asarray(x_data, dtype=float)
		self.v_data = np.asarray(v_data, dtype=float)

		if self.x_data.ndim != 1 or self.v_data.ndim != 1:
			raise ValueError("x_data et v_data doivent etre des tableaux 1D.")
		if self.x_data.size != self.v_data.size:
			raise ValueError("x_data et v_data doivent avoir la meme taille.")
		if self.x_data.size < 2:
			raise ValueError("Au moins deux points de mesure sont requis.")

		self.width_func = width_func if width_func is not None else (lambda x: 0.5 + 0.1 * x)
		self._interpolator = PolynomialInterpolation(self.x_data, self.v_data)
		self._adaptive = AdaptiveIntegration(tol=1e-6, max_depth=25)

		self._x_min = float(np.min(self.x_data))
		self._x_max = float(np.max(self.x_data))

	def velocity(self, x_eval):
		"""Retourne la vitesse interpolee au(x) point(s) x_eval."""

		return self._interpolator.evaluate(x_eval, method="newton")

	def _width(self, x_eval):
		x = np.asarray(x_eval, dtype=float)

		if x.ndim == 0:
			return float(self.width_func(float(x)))

		# Accepte les fonctions vectorisees et non vectorisees.
		try:
			w = np.asarray(self.width_func(x), dtype=float)
			if w.shape == x.shape:
				return w
		except Exception:
			pass

		return np.array([float(self.width_func(float(xi))) for xi in x], dtype=float)

	def local_flow_rate(self, x_eval):
		"""Retourne le debit local q(x) = v(x) * w(x)."""

		return self.velocity(x_eval) * self._width(x_eval)

	def _integrate(self, f, a, b, method="adaptive", n=100):
		method_key = str(method).strip().lower()

		if method_key == "rectangle":
			return NewtonCotes.rectangle(f, a, b, n=n)
		if method_key == "trapezoidal":
			return NewtonCotes.trapezoidal(f, a, b, n=n)
		if method_key == "simpson":
			return NewtonCotes.simpson(f, a, b, n=n)
		if method_key == "adaptive":
			return self._adaptive.adaptive_simpson(f, a, b)
		if method_key == "gauss2":
			return GaussQuadrature.gauss_legendre_2(f, a, b)
		if method_key == "gauss3":
			return GaussQuadrature.gauss_legendre_3(f, a, b)

		raise ValueError(
			"Methode d'integration inconnue. Choisissez parmi: "
			"rectangle, trapezoidal, simpson, adaptive, gauss2, gauss3."
		)

	def total_flow_rate(self, method="adaptive", n=100):
		"""Calcule le debit total sur [min(x_data), max(x_data)]."""

		return self._integrate(lambda x: float(self.local_flow_rate(x)), self._x_min, self._x_max, method=method, n=n)

	def _velocity_derivative(self, x_eval):
		x = np.asarray(x_eval, dtype=float)
		scale = max(self._x_max - self._x_min, 1.0)
		dx = 1e-6 * scale

		if x.ndim == 0:
			x_value = float(x)
			return (float(self.velocity(x_value + dx)) - float(self.velocity(x_value - dx))) / (2.0 * dx)

		return (np.asarray(self.velocity(x + dx), dtype=float) - np.asarray(self.velocity(x - dx), dtype=float)) / (2.0 * dx)

	def acceleration(self, x_eval):
		"""Acceleration convective a(x) = v(x) * dv/dx."""

		return self.velocity(x_eval) * self._velocity_derivative(x_eval)

	def work(self, mass=2.0):
		"""Travail mecanique W = integral(F dx) avec F(x) = m * a(x)."""

		mass = float(mass)
		if mass <= 0:
			raise ValueError("mass doit etre strictement positive.")

		force = lambda x: mass * float(self.acceleration(x))
		return self._adaptive.adaptive_simpson(force, self._x_min, self._x_max)