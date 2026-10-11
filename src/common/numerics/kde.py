"""Gaussian smoothing, continuous divergence metrics and KDE bandwidth policies."""

from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.stats import gaussian_kde, norm


# KDE primitives
def gaussian_kernel(x: NDArray, h: float) -> NDArray:
    """Evaluate the normalized Gaussian kernel K_h at offsets x, with bandwidth h in x units."""
    return np.exp(-0.5 * (x / h) ** 2) / (np.sqrt(2 * np.pi) * h)


def compute_kde_density(x_grid: NDArray, x_data: NDArray, h: float, epsilon: float = 1e-14) -> NDArray:
    """Evaluate the existing smoothing convention sum(K_h(x-x_i))/(N*h) + epsilon on x_grid."""
    diff = x_grid.reshape(-1, 1) - x_data.reshape(1, -1)
    kde = np.sum(gaussian_kernel(diff, h), axis=1)
    return kde / (x_data.size * h) + epsilon


def compute_weighted_stats(
    x_result: NDArray, x_data: NDArray, y_data: NDArray, bandwidths: NDArray, epsilon: float = 1e-14
) -> Tuple[NDArray, NDArray]:
    """Return local weighted means and nonnegative variances; bandwidths has one value per x_result point."""
    diff = x_result.reshape(-1, 1) - x_data.reshape(1, -1)
    weights_unnorm = gaussian_kernel(diff, cast(float, bandwidths.reshape(-1, 1)))
    weights = weights_unnorm / (np.sum(weights_unnorm, axis=1, keepdims=True) + epsilon)

    means = np.sum(weights * y_data.reshape(1, -1), axis=1)
    means2 = np.sum(weights * (y_data.reshape(1, -1) ** 2), axis=1)
    variances = np.maximum(means2 - means**2, 0)

    return means, variances


# Divergence metrics
def kl_divergence_continuous(p_func, q_func, xmin=0, xmax=2, t_err=1e-13) -> float:
    """Integrate KL(p || q) on [xmin, xmax], flooring both densities at t_err."""

    def integrand(x):
        p = max(p_func(x), t_err)
        q = max(q_func(x), t_err)
        return p * (np.log(p) - np.log(q))

    val, _ = quad(integrand, xmin, xmax, limit=100)
    return val


def js_divergence_continuous(p_func, q_func, xmin=0, xmax=2, t_err=1e-13) -> float:
    """Return the square root of integrated Jensen-Shannon divergence with a density floor."""

    def integrand(x):
        p = max(p_func(x), t_err)
        q = max(q_func(x), t_err)
        m = (p + q) / 2
        return 0.5 * (p * (np.log(p) - np.log(m)) + q * (np.log(q) - np.log(m)))

    val, _ = quad(integrand, xmin, xmax, limit=100)
    return np.sqrt(val)


LINSPACE_SMPL_COUNT = 256


def fast_trapz(y: NDArray, x: NDArray):
    """Integrate sampled y over a uniformly spaced x grid using trapezoidal endpoint weights."""
    h = x[1] - x[0]
    result = h * (y.sum() - 0.5 * (y[0] + y[-1]))
    return result


def kl_divergence_continuous_fast(p_func, q_func, xmin=0, xmax=2, n: NDArray | int = LINSPACE_SMPL_COUNT, t_err=1e-13):
    """Approximate KL on n uniform samples; when n is an array, use it as x and accept sampled p_func/q_func arrays."""
    if isinstance(n, np.ndarray):
        xs = n
        p_vals = np.maximum(p_func, t_err)
        q_vals = np.maximum(q_func, t_err)
    else:
        xs = np.linspace(xmin, xmax, n)
        p_vals = np.maximum(p_func(xs), t_err)
        q_vals = np.maximum(q_func(xs), t_err)
    integrand = p_vals * (np.log(p_vals) - np.log(q_vals))
    return np.trapz(integrand, xs)


def js_divergence_continuous_fast(p_func, q_func, xmin=0, xmax=2, n: NDArray | int = LINSPACE_SMPL_COUNT, t_err=1e-13):
    """Approximate Jensen-Shannon distance on uniform samples; array n supplies the grid and sampled densities."""
    if isinstance(n, np.ndarray):
        xs = n
        p_vals = np.maximum(p_func, t_err)
        q_vals = np.maximum(q_func, t_err)
    else:
        xs = np.linspace(xmin, xmax, n)
        p_vals = np.maximum(p_func(xs), t_err)
        q_vals = np.maximum(q_func(xs), t_err)
    m_vals = 0.5 * (p_vals + q_vals)
    integrand = 0.5 * (p_vals * (np.log(p_vals) - np.log(m_vals)) + q_vals * (np.log(q_vals) - np.log(m_vals)))
    val = fast_trapz(integrand, xs)
    return np.sqrt(val)


# KDE bandwidth helpers
def kde_min_bw_factory(min_bandwidth):
    """Return a scipy.gaussian_kde bandwidth callback enforcing a minimum width in data units."""

    def min_bw_factor(kde_obj):
        default_factor = kde_obj.scotts_factor()
        min_factor = min_bandwidth / np.std(kde_obj.dataset, ddof=1)
        return max(default_factor, min_factor)

    return min_bw_factor


def kde_min_bw_calc(data: NDArray, min_bw=0.1):
    """Return the larger of the normal-reference bandwidth and min_bw for a one-dimensional sample."""
    std = np.std(data, ddof=1)
    default_bw = 1.06 * std * len(data) ** (-1 / 5)
    bw = max(default_bw, min_bw)
    return bw


def min_bandwidth_enforcer(bandwidth, min_bandwidth, data):
    """Return the maximum of the requested width, minimum width and one tenth of the sample standard deviation."""
    return max(bandwidth, min_bandwidth, np.std(data, ddof=1) * 0.1)


def get_kde_pdf(data, min_bandwidth: float, xmin: float, xmax: float):
    # A constant sample makes scipy's KDE singular.
    """Return a callable scipy KDE, or a min_bandwidth Gaussian for constant samples; xmin/xmax are unused."""
    if np.all(data == data[0]):
        # Approximate a delta distribution centered on the repeated value.
        def delta_like_pdf(x):
            return norm.pdf(x, data[0], min_bandwidth)

        return delta_like_pdf

    bw_method = kde_min_bw_factory(min_bandwidth)
    data_smpl = data  # np.random.choice(data, size=2000, replace=False)
    kde = gaussian_kde(data_smpl, bw_method=bw_method)
    return kde
