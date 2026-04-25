from __future__ import annotations

import numpy as np


def _as_float(value):
	"""Convertit proprement une valeur scalaire numérique en float (réél)."""

	return float(np.asarray(value, dtype=float))


def _validate_interval(a, b):
	"""Normalise les bornes d'intégration."""

	return _as_float(a), _as_float(b)


def _validate_positive_integer(name, value):
	"""Vérifie qu'une valeur est un entier strictement positif."""

	if isinstance(value, bool):
		raise ValueError(f"{name} doit être un entier strictement positif.")

	try:
		numeric_value = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{name} doit être un entier strictement positif.") from exc

	if not numeric_value.is_integer():
		raise ValueError(f"{name} doit être un entier strictement positif.")

	integer_value = int(numeric_value)
	if integer_value <= 0:
		raise ValueError(f"{name} doit être un entier strictement positif.")
	return integer_value


def _sum_function_values(f, points):
	"""Évalue f sur une suite de points et additionne les résultats."""

	return sum(_as_float(f(point)) for point in points)


class NewtonCotes:
	"""Formules classiques de Newton-Cotes composées."""

	@staticmethod
	def rectangle(f, a, b, n=1):
		"""Intégration par la méthode des rectangles (somme à gauche).

		Args:
			f: Fonction à intégrer.
			a: Borne inférieure.
			b: Borne supérieure.
			n: Nombre de sous-intervalles.

		Returns:
			Approximation de l'intégrale de f sur [a, b].
		"""

		n = _validate_positive_integer("n", n)
		a, b = _validate_interval(a, b)

		if a == b:
			return 0.0

		h = (b - a) / n
		points = a + h * np.arange(n)
		return h * _sum_function_values(f, points)

	@staticmethod
	def trapezoidal(f, a, b, n=1):
		"""Intégration par la méthode des trapèzes composée."""

		n = _validate_positive_integer("n", n)
		a, b = _validate_interval(a, b)

		if a == b:
			return 0.0

		h = (b - a) / n
		interior_points = a + h * np.arange(1, n)
		return h * (
			0.5 * _as_float(f(a))
			+ _sum_function_values(f, interior_points)
			+ 0.5 * _as_float(f(b))
		)

	@staticmethod
	def simpson(f, a, b, n=2):
		"""Intégration par la méthode de Simpson composée.

		Le nombre de sous-intervalles n doit être pair.
		"""

		n = _validate_positive_integer("n", n)
		if n % 2 != 0:
			raise ValueError("La méthode de Simpson nécessite un nombre d'intervalles pair.")

		a, b = _validate_interval(a, b)

		if a == b:
			return 0.0

		h = (b - a) / n
		odd_points = a + h * np.arange(1, n, 2)
		even_points = a + h * np.arange(2, n, 2)

		return (h / 3.0) * (
			_as_float(f(a))
			+ _as_float(f(b))
			+ 4.0 * _sum_function_values(f, odd_points)
			+ 2.0 * _sum_function_values(f, even_points)
		)

	@staticmethod
	def simpson_38(f, a, b, n=3):
		"""Intégration par la méthode de Simpson 3/8 composée.

		Le nombre de sous-intervalles n doit être multiple de 3.
		"""

		n = _validate_positive_integer("n", n)
		if n % 3 != 0:
			raise ValueError("La méthode de Simpson 3/8 nécessite un nombre d'intervalles multiple de 3.")

		a, b = _validate_interval(a, b)

		if a == b:
			return 0.0

		h = (b - a) / n
		interior_indices = np.arange(1, n)
		multiple_of_three = interior_indices[interior_indices % 3 == 0]
		other_points = interior_indices[interior_indices % 3 != 0]

		return (3.0 * h / 8.0) * (
			_as_float(f(a))
			+ _as_float(f(b))
			+ 3.0 * _sum_function_values(f, a + h * other_points)
			+ 2.0 * _sum_function_values(f, a + h * multiple_of_three)
		)


class AdaptiveIntegration:
	"""Intégration adaptative basée sur la règle de Simpson."""

	def __init__(self, tol=1e-6, max_depth=20):
		self.tol = float(tol)
		self.max_depth = _validate_positive_integer("max_depth", max_depth)

		if self.tol <= 0:
			raise ValueError("tol doit être strictement positif.")

	def _simpson_rule(self, f, a, b):
		"""Règle de Simpson sur un seul intervalle."""

		a, b = _validate_interval(a, b)
		c = (a + b) / 2.0
		return (b - a) / 6.0 * (_as_float(f(a)) + 4.0 * _as_float(f(c)) + _as_float(f(b)))

	def adaptive_simpson(self, f, a, b, tol=None, depth=0):
		"""Intégration adaptative par subdivision récursive."""

		a, b = _validate_interval(a, b)
		tol = self.tol if tol is None else float(tol)
		if tol <= 0:
			raise ValueError("tol doit être strictement positif.")

		whole = self._simpson_rule(f, a, b)
		c = (a + b) / 2.0
		left = self._simpson_rule(f, a, c)
		right = self._simpson_rule(f, c, b)
		delta = left + right - whole

		if depth >= self.max_depth:
			return left + right + delta / 15.0

		if abs(delta) <= 15.0 * tol:
			return left + right + delta / 15.0

		return (
			self.adaptive_simpson(f, a, c, tol=tol / 2.0, depth=depth + 1)
			+ self.adaptive_simpson(f, c, b, tol=tol / 2.0, depth=depth + 1)
		)


class GaussQuadrature:
	"""Quadratures de Gauss-Legendre sur un intervalle quelconque."""

	@staticmethod
	def _transform_nodes(a, b, nodes):
		a, b = _validate_interval(a, b)
		midpoint = (a + b) / 2.0
		half_length = (b - a) / 2.0
		return midpoint + half_length * np.asarray(nodes, dtype=float), half_length

	@staticmethod
	def gauss_legendre_2(f, a, b):
		"""Quadrature de Gauss-Legendre à 2 points."""

		nodes, half_length = GaussQuadrature._transform_nodes(a, b, [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
		weights = [1.0, 1.0]
		return half_length * sum(w * _as_float(f(x)) for w, x in zip(weights, nodes))

	@staticmethod
	def gauss_legendre_3(f, a, b):
		"""Quadrature de Gauss-Legendre à 3 points."""

		nodes, half_length = GaussQuadrature._transform_nodes(a, b, [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
		weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
		return half_length * sum(w * _as_float(f(x)) for w, x in zip(weights, nodes))
