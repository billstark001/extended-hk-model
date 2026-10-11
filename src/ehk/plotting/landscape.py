"""Shared numerical and visual helpers for potential landscapes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from common.numerics import estimate_potential_from_force


def _potential_from_force(x: np.ndarray, force: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Integrate one or more force curves along their final axis."""

    x_values = np.asarray(x, dtype=float)
    force_values = np.asarray(force, dtype=float)
    if x_values.ndim != 1 or force_values.shape[-1] != x_values.size:
        raise ValueError("force curve's final axis must match one-dimensional x")
    flat_force = force_values.reshape((-1, x_values.size))
    flat_potential = np.stack([estimate_potential_from_force(x_values, row) for row in flat_force])
    potential = flat_potential.reshape(force_values.shape)
    if center:
        potential -= np.mean(potential, axis=-1, keepdims=True)
    return np.asarray(potential, dtype=float)


CMAP_NAME = "managua"


def potential_from_force(x: np.ndarray, force: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Integrate negative force along its final opinion axis; center=True subtracts each potential curve mean."""
    return _potential_from_force(x, force, center=center)


def format_landscape_axis(
    axis: Axes, *, xlabel: str | None, ylabel: str | None, ylim: tuple[float, float] | None = None
) -> None:
    """Apply the existing paper landscape axis style."""
    axis.set_xlim(-1, 1)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(True, linestyle="--", alpha=0.5)
    if xlabel is None:
        axis.set_xticklabels([])
    else:
        axis.set_xlabel(xlabel)
    if ylabel is None:
        axis.set_yticklabels([])
    else:
        axis.set_ylabel(ylabel)


def plot_time_curves(
    axis: Axes, x: np.ndarray, curves: Sequence[np.ndarray] | np.ndarray, normalized_times: Sequence[float] | np.ndarray
) -> None:
    """Plot normalized-time curves using the original landscape styling."""
    cmap = plt.get_cmap(CMAP_NAME)
    for curve, normalized_time in zip(curves, normalized_times):
        axis.plot(x, curve, color=cmap(float(normalized_time)), linewidth=1, alpha=0.8)


def plot_potential_curves(
    axis: Axes,
    x: np.ndarray,
    potentials: Sequence[np.ndarray] | np.ndarray,
    normalized_times: Sequence[float] | np.ndarray,
    *,
    xlabel: bool = True,
    ylabel: bool = True,
) -> None:
    """Draw the supplied potential curves with normalized-time colors and the standard opinion-axis style."""
    plot_time_curves(axis, x, potentials, normalized_times)
    format_landscape_axis(axis, xlabel=r"$x$" if xlabel else None, ylabel=r"$V(x)$" if ylabel else None)


def plot_normalized_time_colorbar(figure: Figure, axes: Axes | Iterable[Axes], *, pad: float = 0.01):
    """Return a shared landscape-colormap colorbar for normalized time in [0,1] attached to the given axes."""
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    scalar = plt.cm.ScalarMappable(cmap=CMAP_NAME, norm=norm)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axes, orientation="vertical", aspect=30, pad=pad)
    colorbar.set_label(r"$t_n$")
    return colorbar


def set_shared_y_limits(axes: Iterable[Axes], values: Sequence[np.ndarray], *, padding_ratio: float = 0.04) -> None:
    """Set a common padded data range across axes; values must be nonempty finite arrays."""
    lower = min(float(np.min(value)) for value in values)
    upper = max(float(np.max(value)) for value in values)
    padding = padding_ratio * max(upper - lower, 1e-12)
    for axis in axes:
        axis.set_ylim(lower - padding, upper + padding)


def save_landscape_figure(figure: Figure, output_stem: str | Path) -> tuple[Path, Path]:
    """Save the standard PDF/PNG pair without opening an interactive window."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path
