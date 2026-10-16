"""Offline node/edge fitting and figures for one saved parameter slot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.io.json import atomic_json, sha
from common.plotting.figures import decorate, save
from theory.release.models import sn
from theory.release.quantile_fits import fit_exponentiated_weibull, fit_gompertz, fit_rational_quadratic, survival_curve

from ..data import write_csv
from .expanded import config
from .expanded.analyze.curves import slot_descriptors
from .expanded.fit.edge import fit_slot as fit_edge_slot
from .expanded.fit.node import low_parameter_fits, quantile_times
from .expanded.storage import load_slot

MODELS = (
    ("gompertz_hazard_2", "Gompertz (2)", fit_gompertz),
    ("exponentiated_weibull_3", "Exp. Weibull (3)", fit_exponentiated_weibull),
    ("rational_quadratic_bezier_3", "Rational Bezier (3)", fit_rational_quadratic),
)


def analyze_slot(slot):
    """Retain both quantile-model families and the edge/node-survival mixture."""
    descriptors = slot_descriptors(slot)
    edge = fit_edge_slot(slot)
    return {
        "descriptors": descriptors,
        "low_parameter_fits": low_parameter_fits(descriptors),
        "edge_timescale_fits": [edge] if edge else [],
    }


def plot_slot(slot, fitted, output):
    """Plot raw energies, node/edge model overlays and release-quantile fits."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    tau = slot["tau"]
    node_fit = next(
        (r for r in fitted["low_parameter_fits"] if r["energy"] == "node_energy" and r["model"] == "gompertz_hazard_2"),
        None,
    )
    for i, name in enumerate(config.ENERGY_NAMES):
        raw, quantile = axes[i]
        raw.plot(tau, slot[name], color="black", label="Observed")
        if name == "node_energy" and node_fit:
            _, plateau, amplitude = survival_curve(slot[name])
            predicted = sn(tau, node_fit["initial_hazard"], node_fit["acceleration"])
            raw.plot(tau, plateau + amplitude * predicted, "--", label="Gompertz fit")
        if name == "edge_energy" and fitted["edge_timescale_fits"]:
            edge = fitted["edge_timescale_fits"][0]
            observed_node = survival_curve(slot["node_energy"])[0]
            prediction = (
                observed_node
                if slot["q_over_alpha"] == 0
                else (
                    edge["weight"] * np.exp(-((edge["kappa"] * slot["q_over_alpha"] * tau) ** edge["beta"]))
                    + (1 - edge["weight"]) * observed_node
                )
            )
            raw.plot(
                tau, edge["edge_plateau"] + edge["edge_drop_amplitude"] * prediction, "--", label="Edge mixture fit"
            )
        raw.set(xlabel=r"Scaled time $\tau=\alpha t$", ylabel=name.replace("_", " "))
        raw.legend(frameon=False, fontsize=8)
        row = next(r for r in fitted["descriptors"] if r["energy"] == name)
        times = quantile_times(row)
        quantile.plot(config.PROGRESS_LEVELS, times, "ko", label="Observed quantiles", ms=4)
        if np.all(np.isfinite(times)) and np.all(times > 0):
            for _, label, fit in MODELS:
                prediction, _ = fit(times, config.PROGRESS_LEVELS)
                quantile.plot(config.PROGRESS_LEVELS, prediction, label=label)
        quantile.set(xlabel="Released fraction", ylabel=r"Release time $\tau_p$")
        quantile.legend(frameon=False, fontsize=8)
    decorate(axes)
    fig.suptitle(
        f"{slot['dynamics'].upper()}; epsilon={slot['epsilon']:g}, alpha={slot['alpha']:g}, r={slot['q_over_alpha']:g}\n"
        f"Observed window tau={tau[-1]:g}; saved convergence flag={slot['converged']}; empirical endpoint-referenced fits",
        fontsize=11,
    )
    stem = f"slot_{slot['dynamics']}_e{slot['epsilon']:.12g}_a{slot['alpha']:.12g}_r{slot['q_over_alpha']:.12g}"
    save(fig, Path(output) / stem)
    return stem


def export_slot(source, output, *, dynamics, alpha, epsilon, ratio, plot=False):
    """Analyze one selected archive slot and optionally render PDF/PNG fits."""
    slot, source_path = load_slot(source, dynamics=dynamics, alpha=alpha, epsilon=epsilon, ratio=ratio)
    fitted = analyze_slot(slot)
    output = Path(output).expanduser().resolve()
    if output == source_path.parent or output in source_path.parents:
        raise ValueError("slot exports require a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in fitted.items():
        path = output / (name + ".csv")
        if rows:
            write_csv(path, rows)
        else:
            path.unlink(missing_ok=True)
    figure = plot_slot(slot, fitted, output) if plot else None
    manifest = {
        "schema": 1,
        "source": str(source_path),
        "source_sha256": sha(source_path),
        "slot": {k: slot[k] for k in ("dynamics", "epsilon", "alpha", "q_over_alpha", "q", "converged")},
        "horizon_tau": float(slot["tau"][-1]),
        "time_units": "tau=alpha*t",
        "figure": figure,
        "scope": "offline empirical fits to an observed finite window; no simulation or scan design",
    }
    atomic_json(output / "slot_manifest.json", manifest)
    print(f"Energy slot fitted at {output}", flush=True)
    return manifest
