from __future__ import annotations

import numpy as np

from core.numeric_integration import AdaptiveIntegration, GaussQuadrature, NewtonCotes
from core.polynomial_interpolation import PolynomialInterpolation


class CoolingProblem:
	"""Modelise le probleme de refroidissement d'un composant."""

	def __init__(self, t_data, T_data, T_ambient=20.0, h_coeff=50.0):
		"""Initialise les donnees, l'interpolateur et les constantes thermiques."""

		self.t_data = np.asarray(t_data, dtype=float)
		self.T_data = np.asarray(T_data, dtype=float)

		if self.t_data.ndim != 1 or self.T_data.ndim != 1:
			raise ValueError("t_data et T_data doivent etre des tableaux 1D.")
		if self.t_data.size != self.T_data.size:
			raise ValueError("t_data et T_data doivent avoir la meme taille.")
		if self.t_data.size < 2:
			raise ValueError("Au moins deux points de mesure sont requis.")

		self.T_ambient = float(T_ambient)
		self.h_coeff = float(h_coeff)
		if self.h_coeff <= 0:
			raise ValueError("h_coeff doit etre strictement positif.")

		self._interpolator = PolynomialInterpolation(self.t_data, self.T_data)
		self._adaptive = AdaptiveIntegration(tol=1e-6, max_depth=25)

		self._t_min = float(np.min(self.t_data))
		self._t_max = float(np.max(self.t_data))
		self._t_ref = float(self.t_data[np.argmin(self.t_data)])
		self._T_ref = float(self.T_data[np.argmin(self.t_data)])

	def temperature(self, t_eval):
		"""Retourne la temperature interpolee au(x) instant(s) t_eval."""

		return self._interpolator.evaluate(t_eval, method="newton")

	def heat_loss_rate(self, t_eval):
		"""Retourne le flux thermique h * (T(t) - T_ambient)."""

		return self.h_coeff * (self.temperature(t_eval) - self.T_ambient)

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

	def total_heat_loss(self, method="adaptive", n=100):
		"""Calcule la chaleur totale dissipee entre min(t_data) et max(t_data)."""

		return self._integrate(lambda t: float(self.heat_loss_rate(t)), self._t_min, self._t_max, method=method, n=n)

	def exponential_model(self, t_eval, k):
		"""Modele exponentiel: T(t) = Tamb + (T0 - Tamb) * exp(-k * (t - t0))."""

		k = float(k)
		if k < 0:
			raise ValueError("k doit etre positif ou nul.")

		t = np.asarray(t_eval, dtype=float)
		values = self.T_ambient + (self._T_ref - self.T_ambient) * np.exp(-k * (t - self._t_ref))
		if t.ndim == 0:
			return float(values)
		return values

	def model_error(self, k):
		"""Erreur quadratique moyenne entre le modele exponentiel et les mesures."""

		model_values = np.asarray(self.exponential_model(self.t_data, k), dtype=float)
		return float(np.mean((model_values - self.T_data) ** 2))

	def estimate_k(self, k_min=0.01, k_max=0.5, tol=1e-4):
		"""Estime k via recherche par section doree sur [k_min, k_max]."""

		left = float(k_min)
		right = float(k_max)
		tol = float(tol)

		if left <= 0 or right <= 0 or left >= right:
			raise ValueError("Il faut 0 < k_min < k_max.")
		if tol <= 0:
			raise ValueError("tol doit etre strictement positif.")

		phi = (1.0 + np.sqrt(5.0)) / 2.0
		x1 = right - (right - left) / phi
		x2 = left + (right - left) / phi
		f1 = self.model_error(x1)
		f2 = self.model_error(x2)

		while (right - left) > tol:
			if f1 > f2:
				left = x1
				x1 = x2
				f1 = f2
				x2 = left + (right - left) / phi
				f2 = self.model_error(x2)
			else:
				right = x2
				x2 = x1
				f2 = f1
				x1 = right - (right - left) / phi
				f1 = self.model_error(x1)

		return 0.5 * (left + right)