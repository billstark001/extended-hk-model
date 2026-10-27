"""CSV-only diagnostic renderer; no solver, raw reader or estimator imports."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from common.experiments.lifecycle import session_lock
from common.io.json import atomic_json, sha
from common.plotting.core import plt_save_and_close

INPUTS = ("probabilities.csv", "moments.csv", "cdf_bounds.csv", "closure_error.csv")
FIGURES = ("conditional_probabilities", "joint_noise", "commitment_times", "closure_accuracy")


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def plot(directory):
    directory = Path(directory)
    with session_lock(directory):
        inputs = {name: sha(directory / name) for name in INPUTS}
        manifest = directory / "manifest.json"
        if manifest.exists():
            analyzed = json.loads(manifest.read_text())
            if any(analyzed["files"].get(name) != digest for name, digest in inputs.items()):
                raise ValueError("analysis tables changed; analyze again before plotting")
        _plot(directory)
        if any(sha(directory / name) != digest for name, digest in inputs.items()):
            raise ValueError("plot input changed before publication")
        atomic_json(
            directory / "plot_manifest.json",
            dict(
                analysis_manifest_sha256=sha(manifest) if manifest.exists() else None,
                inputs=inputs,
                files={
                    name + suffix: sha(directory / (name + suffix)) for name in FIGURES for suffix in (".png", ".pdf")
                },
            ),
        )


def _plot(directory):
    data = rows(directory / "probabilities.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for kind, color in (("micro_probe", "#4477aa"), ("meso_probe", "#ee7733")):
        chosen = [
            r
            for r in data
            if r["kind"] == kind
            and r["variant"] == "stochastic"
            and r["layer"] == "naive"
            and (int(r["paths"]) > int(r["pending_paths"]) or r["probability_method"] == "terminal_boundary")
        ]
        axes[0].scatter(
            [float(r["tau"]) for r in chosen],
            [float(r["estimate_1"]) for r in chosen],
            s=12,
            label=kind,
            color=color,
            alpha=0.65,
        )
        axes[1].scatter(
            [float(r["tau"]) for r in chosen],
            [float(r["disagreement_lower"]) for r in chosen],
            s=12,
            color=color,
            alpha=0.65,
        )
    axes[0].set(xlabel="Query time tau", ylabel="Consensus probability estimate", ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Query time tau", ylabel="Outcome disagreement (allocation lower bound)")
    plt_save_and_close(fig, str(directory / "conditional_probabilities"))
    data = rows(directory / "moments.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for kind, color in (("micro_moments", "#4477aa"), ("meso_moments", "#ee7733")):
        selected = [r for r in data if r["kind"] == kind and r["layer"] == "naive"]
        for ax, key in zip(axes, ("node_variance_trace", "edge_variance_trace")):
            ax.scatter(
                [float(r["bins"]) for r in selected], [float(r[key]) for r in selected], s=15, label=kind, color=color
            )
            ax.set(xlabel="Opinion bins", ylabel=key.replace("_", " "))
    axes[0].legend(fontsize=8)
    plt_save_and_close(fig, str(directory / "joint_noise"))
    data = rows(directory / "cdf_bounds.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for ratio in sorted({float(r["ratio"]) for r in data}):
        selected = sorted(
            (r for r in data if float(r["ratio"]) == ratio and float(r["delta"]) == 0.1), key=lambda r: float(r["tau"])
        )
        x = [float(r["tau"]) for r in selected]
        lo, hi = [float(r["cdf_lower"]) for r in selected], [float(r["cdf_upper"]) for r in selected]
        line = ax.plot(x, lo, label=f"r={ratio:g}")[0]
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.15)
    ax.set(xlabel="Observation-grid time tau", ylabel="Commitment CDF bounds (delta=0.1)", ylim=(-0.02, 1.02))
    if data:
        ax.legend(fontsize=8)
    plt_save_and_close(fig, str(directory / "commitment_times"))
    data = rows(directory / "closure_error.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for variant in sorted({r["variant"] for r in data}):
        selected = [r for r in data if r["variant"] == variant]
        ax.scatter([float(r["bins"]) for r in selected], [float(r["tv"]) for r in selected], s=18, label=variant)
    ax.set(xlabel="Opinion bins", ylabel="Conditional outcome TV", ylim=(-0.02, 1.02))
    if data:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No completely resolved matched distributions", ha="center", transform=ax.transAxes)
    plt_save_and_close(fig, str(directory / "closure_accuracy"))
