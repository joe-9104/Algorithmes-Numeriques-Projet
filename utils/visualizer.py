from __future__ import annotations

from collections.abc import Callable, Mapping

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

	def plot_interpolation_comparison(self, x_data, y_data, interpolators, x_fine, title):
		"""Plot data points and several interpolation curves on the same figure."""

		x_data = self._as_array(x_data)
		y_data = self._as_array(y_data)
		x_fine = self._as_array(x_fine)

		if x_data.shape != y_data.shape:
			raise ValueError("x_data and y_data must have the same shape.")

		fig, ax = plt.subplots(figsize=self.figsize)
		ax.scatter(x_data, y_data, color="black", marker="o", s=40, label="Data")

		if isinstance(interpolators, Mapping):
			items = interpolators.items()
		else:
			items = [(f"Interpolator {i + 1}", interp) for i, interp in enumerate(interpolators)]

		for name, interp in items:
			y_curve = self._eval_interpolator(interp, x_fine)
			ax.plot(x_fine, y_curve, linewidth=2, label=str(name))

		ax.set_title(str(title))
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.legend()
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		return fig, ax

	def plot_runge_phenomenon(self, x_fine, y_true, interpolations):
		"""Plot Runge target function and interpolation approximations."""

		x_fine = self._as_array(x_fine)
		y_true = self._as_array(y_true)

		fig, ax = plt.subplots(figsize=self.figsize)
		ax.plot(x_fine, y_true, color="black", linewidth=2.5, label="True function")

		if isinstance(interpolations, Mapping):
			items = interpolations.items()
		else:
			items = [(f"Approximation {i + 1}", values) for i, values in enumerate(interpolations)]

		for label, values in items:
			y_values = self._as_array(values)
			ax.plot(x_fine, y_values, linewidth=1.8, label=str(label))

		ax.set_title("Runge Phenomenon")
		ax.set_xlabel("x")
		ax.set_ylabel("f(x)")
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
		ax.set_xlabel("n")
		ax.set_ylabel("Absolute error")
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
		ax.scatter(t_data, T_data, color="black", marker="o", s=45, label="Experimental data")

		if isinstance(T_interp, Mapping):
			for label, values in T_interp.items():
				ax.plot(t_fine, self._as_array(values), linewidth=2, label=f"Interpolation: {label}")
		else:
			ax.plot(t_fine, self._as_array(T_interp), linewidth=2, label="Interpolation")

		ax.plot(t_fine, T_model, linestyle="--", linewidth=2.2, label=f"Exponential model (k={float(k_opt):.4f})")

		ax.set_title("Cooling Analysis")
		ax.set_xlabel("Time (s)")
		ax.set_ylabel("Temperature")
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
		ax1.scatter(x_data, v_data, color="black", marker="o", s=40, label="Measured velocity")

		if isinstance(v_interp, Mapping):
			for label, values in v_interp.items():
				ax1.plot(x_fine, self._as_array(values), linewidth=2, label=f"Velocity: {label}")
		else:
			ax1.plot(x_fine, self._as_array(v_interp), linewidth=2, label="Interpolated velocity")

		ax1.set_xlabel("x (m)")
		ax1.set_ylabel("Velocity (m/s)")

		try:
			widths = np.asarray(w_function(x_fine), dtype=float)
			if widths.shape != x_fine.shape:
				raise ValueError
		except Exception:
			widths = np.array([float(w_function(float(x))) for x in x_fine], dtype=float)

		ax2 = ax1.twinx()
		ax2.plot(x_fine, widths, color="tab:green", linestyle="--", linewidth=2, label="Channel width")
		ax2.set_ylabel("Width (m)")

		handles1, labels1 = ax1.get_legend_handles_labels()
		handles2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

		ax1.set_title("Flow Analysis")
		ax1.grid(True, alpha=0.3)
		fig.tight_layout()
		return fig, (ax1, ax2)