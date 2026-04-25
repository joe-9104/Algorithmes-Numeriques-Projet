from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.numeric_integration import NewtonCotes
from core.polynomial_interpolation import PolynomialInterpolation
from models.flow import FlowProblem
from models.temperatures import CoolingProblem
from utils.visualizer import Visualizer


def get_cooling_data():
	"""Return experimental cooling data from the project statement."""

	t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
	T_data = np.array([90, 85, 72, 63, 58, 52, 48, 45, 43, 41, 40], dtype=float)
	return t_data, T_data


def get_flow_data():
	"""Return flow velocity data and channel width law from the project statement."""

	x_data = np.array([0, 0.5, 1.2, 1.8, 2.5, 3.1, 3.7, 4.2, 4.8, 5.3, 6.0], dtype=float)
	v_data = np.array([0, 2.1, 3.8, 5.2, 6.4, 7.0, 7.3, 7.2, 6.8, 5.9, 4.5], dtype=float)
	width_func = lambda x: 0.5 + 0.1 * np.asarray(x, dtype=float)
	return x_data, v_data, width_func


def runge_function(x):
	"""Runge function used in the analysis section."""

	x = np.asarray(x, dtype=float)
	return 1.0 / (1.0 + 25.0 * x**2)


def chebyshev_nodes(n, a, b):
	"""Return n Chebyshev nodes mapped to [a, b]."""

	k = np.arange(1, n + 1, dtype=float)
	x_unit = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n))
	return (a + b) / 2.0 + (b - a) / 2.0 * x_unit


def save_figure(fig, output_dir, name):
	"""Persist a Matplotlib figure under output_dir."""

	path = output_dir / name
	fig.savefig(path, dpi=180, bbox_inches="tight")
	return path


def run_runge_study(viz, output_dir):
	"""Study Runge phenomenon for equidistant vs Chebyshev nodes."""

	a, b = -5.0, 5.0
	x_fine = np.linspace(a, b, 800)
	y_true = runge_function(x_fine)

	interpolation_curves = {}
	max_errors = {}

	for n in (5, 10, 15, 20):
		x_eq = np.linspace(a, b, n)
		x_ch = chebyshev_nodes(n, a, b)

		y_eq = runge_function(x_eq)
		y_ch = runge_function(x_ch)

		interp_eq = PolynomialInterpolation(x_eq, y_eq)
		interp_ch = PolynomialInterpolation(x_ch, y_ch)

		y_eq_fine = np.asarray(interp_eq.evaluate(x_fine, method="newton"), dtype=float)
		y_ch_fine = np.asarray(interp_ch.evaluate(x_fine, method="newton"), dtype=float)

		interpolation_curves[f"Equidistant n={n}"] = y_eq_fine
		interpolation_curves[f"Chebyshev n={n}"] = y_ch_fine
		max_errors[f"eq_n_{n}"] = float(np.max(np.abs(y_eq_fine - y_true)))
		max_errors[f"ch_n_{n}"] = float(np.max(np.abs(y_ch_fine - y_true)))

	fig, _ = viz.plot_runge_phenomenon(x_fine, y_true, interpolation_curves)
	saved = save_figure(fig, output_dir, "runge_phenomenon.png")

	return {
		"figure": str(saved),
		"max_errors": max_errors,
	}


def run_cooling_analysis(viz, output_dir):
	"""Analyze cooling problem with interpolation, integration, and model fitting."""

	t_data, T_data = get_cooling_data()
	cooling = CoolingProblem(t_data, T_data, T_ambient=20.0, h_coeff=50.0)

	t_targets = np.array([2.5, 7.3], dtype=float)
	T_targets = np.asarray(cooling.temperature(t_targets), dtype=float)

	heat_methods = ["rectangle", "trapezoidal", "simpson", "adaptive", "gauss2", "gauss3"]
	total_heat = {
		method: float(cooling.total_heat_loss(method=method, n=100)) for method in heat_methods
	}

	k_opt = float(cooling.estimate_k(k_min=0.01, k_max=0.5, tol=1e-5))

	t_fine = np.linspace(float(t_data.min()), float(t_data.max()), 500)
	poly = PolynomialInterpolation(t_data, T_data)
	T_interp = {
		"newton": np.asarray(poly.evaluate(t_fine, method="newton"), dtype=float),
		"lagrange": np.asarray(poly.evaluate(t_fine, method="lagrange"), dtype=float),
	}
	T_model = np.asarray(cooling.exponential_model(t_fine, k_opt), dtype=float)

	fig_cmp, _ = viz.plot_interpolation_comparison(
		t_data,
		T_data,
		{
			"Newton": lambda x: poly.evaluate(x, method="newton"),
			"Lagrange": lambda x: poly.evaluate(x, method="lagrange"),
		},
		t_fine,
		"Cooling Interpolation Comparison",
	)
	fig_analysis, _ = viz.plot_cooling_analysis(t_data, T_data, t_fine, T_interp, k_opt, T_model)

	saved_cmp = save_figure(fig_cmp, output_dir, "cooling_interpolation_comparison.png")
	saved_analysis = save_figure(fig_analysis, output_dir, "cooling_analysis.png")

	return {
		"temperature_at_targets": {
			"t_2.5": float(T_targets[0]),
			"t_7.3": float(T_targets[1]),
		},
		"total_heat_loss": total_heat,
		"k_opt": k_opt,
		"figures": [str(saved_cmp), str(saved_analysis)],
	}


def run_flow_analysis(viz, output_dir):
	"""Analyze flow problem with interpolation, flow-rate integration and work."""

	x_data, v_data, width_func = get_flow_data()
	flow = FlowProblem(x_data, v_data, width_func=width_func)

	flow_methods = ["rectangle", "trapezoidal", "simpson", "adaptive", "gauss2", "gauss3"]
	total_flow = {
		method: float(flow.total_flow_rate(method=method, n=100)) for method in flow_methods
	}

	x_fine = np.linspace(float(x_data.min()), float(x_data.max()), 500)
	v_interp = {
		"newton": np.asarray(flow.velocity(x_fine), dtype=float),
	}

	fig, _ = viz.plot_flow_analysis(x_data, v_data, x_fine, v_interp, width_func)
	saved = save_figure(fig, output_dir, "flow_analysis.png")

	return {
		"total_flow_rate": total_flow,
		"acceleration_at_3.0": float(flow.acceleration(3.0)),
		"work_mass_2.0": float(flow.work(mass=2.0)),
		"figure": str(saved),
	}


def run_convergence_study(viz, output_dir):
	"""Study convergence on f(x)=exp(x) over [0,1]."""

	exact = float(np.e - 1.0)
	n_values = np.array([2, 4, 8, 16, 32, 64, 128, 256], dtype=int)
	f = np.exp

	errors = {
		"rectangle": [],
		"trapezoidal": [],
		"simpson": [],
	}

	for n in n_values:
		errors["rectangle"].append(abs(NewtonCotes.rectangle(f, 0.0, 1.0, n=n) - exact))
		errors["trapezoidal"].append(abs(NewtonCotes.trapezoidal(f, 0.0, 1.0, n=n) - exact))
		errors["simpson"].append(abs(NewtonCotes.simpson(f, 0.0, 1.0, n=n) - exact))

	fig, _ = viz.plot_convergence(
		n_values,
		errors,
		methods=["rectangle", "trapezoidal", "simpson"],
		title="Integration Convergence on exp(x)",
	)
	saved = save_figure(fig, output_dir, "integration_convergence.png")

	return {
		"exact_integral": exact,
		"n_values": n_values.tolist(),
		"errors": {key: [float(value) for value in vals] for key, vals in errors.items()},
		"figure": str(saved),
	}


def main():
	"""Execute all project analyses and generate outputs."""

	parser = argparse.ArgumentParser(description="Projet d'analyse numerique - programme principal")
	parser.add_argument("--output-dir", default="results", help="Directory where plots and results are saved")
	parser.add_argument("--no-show", action="store_true", help="Do not display matplotlib windows")
	args = parser.parse_args()

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	viz = Visualizer(style="seaborn-v0_8-darkgrid", figsize=(10, 6))

	report = {
		"runge": run_runge_study(viz, output_dir),
		"cooling": run_cooling_analysis(viz, output_dir),
		"flow": run_flow_analysis(viz, output_dir),
		"convergence": run_convergence_study(viz, output_dir),
	}

	report_path = output_dir / "summary.json"
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

	print("Execution complete.")
	print(f"Results directory: {output_dir.resolve()}")
	print(f"Summary file: {report_path.resolve()}")

	if args.no_show:
		plt.close("all")
	else:
		try:
			plt.show()
		except Exception:
			# In headless environments, showing figures may fail; files are already saved.
			plt.close("all")


if __name__ == "__main__":
	main()
