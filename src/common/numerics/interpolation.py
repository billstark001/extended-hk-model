"""Shape-preserving interpolation independent of study configuration."""

from scipy.interpolate import PchipInterpolator


def pchip(*args, **kwargs):
    """Construct a piecewise cubic monotone interpolant."""
    return PchipInterpolator(*args, **kwargs)
