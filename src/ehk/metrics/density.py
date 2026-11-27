"""Exact and linearly binned Gaussian evaluation on the metric's original grid."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft

BINNED_POINTS = 4081


@dataclass(frozen=True, eq=False)
class FittedDensity:
    samples: NDArray
    pdf: Callable
    bandwidth: float
    constant: bool


def linear_binned_gaussian(samples, axis, bandwidth, bins=BINNED_POINTS, *, diagnostics=None):
    """Bin mass linearly and convolve with a zero-padded, untruncated Gaussian.

    B-1 must be a multiple of the original grid's 255 intervals, so every
    original evaluation location is embedded exactly in the fine grid.
    """
    if type(bins) is not int or bins < 256 or (bins - 1) % 255:
        raise ValueError("bins - 1 must be a positive multiple of the original 255 intervals")
    if axis.shape != (256,) or not np.array_equal(axis, np.linspace(axis[0], axis[-1], len(axis))):
        raise ValueError("the original 256-point evenly spaced axis is required")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Gaussian bandwidth must be finite and positive")
    data = np.asarray(samples)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("binning requires a nonempty finite one-dimensional sample")
    if np.any(data < axis[0]) or np.any(data > axis[-1]):
        raise ValueError("sample lies outside the padded metric axis")

    stride = (bins - 1) // 255
    spacing = (axis[-1] - axis[0]) / (bins - 1)
    positions = np.clip((data.astype(np.float64) - axis[0]) / spacing, 0, bins - 1)
    left = np.minimum(np.floor(positions).astype(np.int64), bins - 2)
    fraction = positions - left
    mass = np.bincount(left, weights=1 - fraction, minlength=bins)
    mass += np.bincount(left + 1, weights=fraction, minlength=bins)
    mass /= len(data)

    offsets = np.arange(-(bins - 1), bins, dtype=np.float64) * spacing
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2) / (np.sqrt(2 * np.pi) * bandwidth)
    length = next_fast_len(3 * bins - 2, real=True)
    convolved = irfft(rfft(mass, length) * rfft(kernel, length), length)
    fine = convolved[bins - 1:2 * bins - 1]
    minimum = float(np.min(fine))
    roundoff = 128 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(fine))))
    if minimum < -roundoff:
        raise ValueError("FFT density is negative beyond floating-point roundoff")
    if diagnostics is not None:
        diagnostics.update(negative_count=int(np.count_nonzero(fine < 0)), minimum_before_clip=minimum, fft_length=int(length))
    return np.maximum(fine, 0)[::stride]


def evaluate_density(density: FittedDensity | None, axis, *, method, bins=BINNED_POINTS, diagnostics=None):
    """Evaluate only the observed density; constant samples keep their original PDF."""
    if method not in ("exact", "binned"):
        raise ValueError("density method must be 'exact' or 'binned'")
    if density is None:
        if diagnostics is not None:
            diagnostics.update(negative_count=0, minimum_before_clip=None, fft_length=0)
        return np.full_like(axis, np.nan)
    if method == "exact" or density.constant:
        values = density.pdf(axis)
        if diagnostics is not None:
            diagnostics.update(negative_count=0, minimum_before_clip=float(np.min(values)), fft_length=0)
        return values
    return linear_binned_gaussian(density.samples, axis, density.bandwidth, bins, diagnostics=diagnostics)
