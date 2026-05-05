from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
	"""Centralize plotting utilities used in the project."""

	def __init__(self, style="seaborn-v0_8-darkgrid", figsize=(10, 6)):
		"""Initialize plotting style and default figure size."""

		self.style = style
		self.figsize = figsize

		try:
			plt.style.use(self.style)
		except OSError:
			# Fallback to Matplotlib default style if requested style is unavailable.
			plt.style.use("default")

	@staticmethod
	def _as_array(values):
		"""Convert values to a 1D float numpy array."""

		return np.asarray(values, dtype=float)

	@staticmethod
	def _eval_interpolator(interpolator, x_fine):
		"""Evaluate an interpolator object/callable on x_fine."""

		if hasattr(interpolator, "evaluate"):
			return np.asarray(interpolator.evaluate(x_fine), dtype=float)
		if callable(interpolator):
			return np.asarray(interpolator(x_fine), dtype=float)

		raise TypeError("Each interpolator must be callable or implement evaluate(x).")

	@staticmethod
	def _curve_style(index):
		"""Return a deterministic visual style for a curve in a multi-curve plot."""

		styles = [
			{"linestyle": "-", "linewidth": 2.4, "zorder": 3},
			{"linestyle": "--", "linewidth": 2.2, "zorder": 4},
			{"linestyle": ":", "linewidth": 2.2, "zorder": 5},
			{"linestyle": "-.", "linewidth": 2.2, "zorder": 6},
		]
		return styles[index % len(styles)]

	def plot_interpolation_comparison(self, x_data, y_data, interpolators, x_fine, title):
		"""Plot data points and several interpolation curves on the same figure."""

		x_data = self._as_array(x_data)
		y_data = self._as_array(y_data)
		x_fine = self._as_array(x_fine)

		if x_data.shape != y_data.shape:
			raise ValueError("x_data and y_data must have the same shape.")

		fig, ax = plt.subplots(figsize=self.figsize)
		ax.scatter(x_data, y_data, color="black", marker="o", s=40, label="Données expérimentales")

		if isinstance(interpolators, Mapping):
			items = interpolators.items()
		else:
			items = [(f"Interpolator {i + 1}", interp) for i, interp in enumerate(interpolators)]

		for idx, (name, interp) in enumerate(items):
			y_curve = self._eval_interpolator(interp, x_fine)
			style = self._curve_style(idx)
			ax.plot(x_fine, y_curve, label=str(name), **style)

		ax.set_title(str(title))
		ax.set_xlabel("Abscisse")
		ax.set_ylabel("Ordonnée")
		ax.legend()
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		return fig, ax

	def plot_runge_phenomenon(self, x_fine, y_true, interpolations):
		"""Plot Runge target function and interpolation approximations."""

		x_fine = self._as_array(x_fine)
		y_true = self._as_array(y_true)

		fig, ax = plt.subplots(figsize=self.figsize)
		ax.plot(x_fine, y_true, color="black", linewidth=2.5, label="Fonction exacte")

		if isinstance(interpolations, Mapping):
			items = interpolations.items()
		else:
			items = [(f"Approximation {i + 1}", values) for i, values in enumerate(interpolations)]

		for idx, (label, values) in enumerate(items):
			y_values = self._as_array(values)
			style = self._curve_style(idx)
			ax.plot(x_fine, y_values, label=str(label), **style)

		ax.set_title("Phénomène de Runge")
		ax.set_xlabel("Abscisse x")
		ax.set_ylabel("Valeur de f(x)")
		ax.legend()
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		return fig, ax

	def plot_convergence(self, n_values, errors, methods, title):
		"""Plot log-log convergence curves for integration methods."""

		n_values = self._as_array(n_values)
		fig, ax = plt.subplots(figsize=self.figsize)

		if isinstance(errors, Mapping):
			for method in methods:
				method_name = str(method)
				if method_name not in errors:
					raise ValueError(f"Missing error values for method: {method_name}")
				method_errors = self._as_array(errors[method_name])
				ax.loglog(n_values, method_errors, marker="o", linewidth=1.8, label=method_name)
		else:
			errors_array = np.asarray(errors, dtype=float)
			if errors_array.shape[0] != len(methods):
				raise ValueError("errors must contain one sequence per method.")
			for idx, method in enumerate(methods):
				ax.loglog(n_values, errors_array[idx], marker="o", linewidth=1.8, label=str(method))

		ax.set_title(str(title))
		ax.set_xlabel("Nombre d'intervalles n")
		ax.set_ylabel("Erreur absolue")
		ax.legend()
		ax.grid(True, which="both", alpha=0.3)
		fig.tight_layout()
		return fig, ax

	def plot_cooling_analysis(self, t_data, T_data, t_fine, T_interp, k_opt, T_model):
		"""Plot cooling measurements, interpolation(s), and fitted exponential model."""

		t_data = self._as_array(t_data)
		T_data = self._as_array(T_data)
		t_fine = self._as_array(t_fine)
		T_model = self._as_array(T_model)

		if t_data.shape != T_data.shape:
			raise ValueError("t_data and T_data must have the same shape.")

		fig, ax = plt.subplots(figsize=self.figsize)
		ax.scatter(t_data, T_data, color="black", marker="o", s=45, label="Données expérimentales")

		if isinstance(T_interp, Mapping):
			for idx, (label, values) in enumerate(T_interp.items()):
				style = self._curve_style(idx)
				ax.plot(t_fine, self._as_array(values), label=f"Interpolation : {label}", **style)
		else:
			ax.plot(t_fine, self._as_array(T_interp), linewidth=2.2, label="Interpolation", color="tab:blue")

		ax.plot(
			t_fine,
			T_model,
			linestyle="-.",
			linewidth=2.2,
			color="tab:red",
			label=f"Modèle exponentiel (k={float(k_opt):.4f})",
		)

		ax.set_title("Analyse du refroidissement")
		ax.set_xlabel("Temps (s)")
		ax.set_ylabel("Température (°C)")
		ax.legend()
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		return fig, ax

	def plot_flow_analysis(self, x_data, v_data, x_fine, v_interp, w_function):
		"""Plot flow velocity interpolation and channel width profile."""

		x_data = self._as_array(x_data)
		v_data = self._as_array(v_data)
		x_fine = self._as_array(x_fine)

		if x_data.shape != v_data.shape:
			raise ValueError("x_data and v_data must have the same shape.")
		if not callable(w_function):
			raise TypeError("w_function must be callable.")

		fig, ax1 = plt.subplots(figsize=self.figsize)
		ax1.scatter(x_data, v_data, color="black", marker="o", s=40, label="Vitesse mesurée")

		if isinstance(v_interp, Mapping):
			for idx, (label, values) in enumerate(v_interp.items()):
				style = self._curve_style(idx)
				ax1.plot(x_fine, self._as_array(values), label=f"Vitesse interpolée : {label}", **style)
		else:
			ax1.plot(x_fine, self._as_array(v_interp), linewidth=2.2, label="Vitesse interpolée", color="tab:blue")

		ax1.set_xlabel("Position x (m)")
		ax1.set_ylabel("Vitesse (m/s)")

		try:
			widths = np.asarray(w_function(x_fine), dtype=float)
			if widths.shape != x_fine.shape:
				raise ValueError
		except Exception:
			widths = np.asarray([w_function(float(x_val)) for x_val in x_fine.tolist()], dtype=float)

		ax2 = ax1.twinx()
		ax2.plot(x_fine, widths, color="tab:green", linestyle="--", linewidth=2, label="Largeur du canal")
		ax2.set_ylabel("Largeur (m)")

		handles1, labels1 = ax1.get_legend_handles_labels()
		handles2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

		ax1.set_title("Analyse de l’écoulement")
		ax1.grid(True, alpha=0.3)
		fig.tight_layout()
		return fig, (ax1, ax2)