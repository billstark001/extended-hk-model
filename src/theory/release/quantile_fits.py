"""Quantile regression models independent of study designs."""

import numpy as np
from scipy.optimize import least_squares


def fit_gompertz(times: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit positive initial hazard/acceleration by log-time residuals; return predicted quantiles and physical parameters."""
    transformed = -np.log1p(-levels)

    def predict(parameters: np.ndarray) -> np.ndarray:
        initial_hazard, acceleration = np.exp(parameters)
        return np.log1p(acceleration * transformed / initial_hazard) / acceleration

    fit = least_squares(
        lambda parameters: np.log(predict(parameters)) - np.log(times),
        np.log((0.1, 0.3)),
        bounds=((-15.0, -15.0), (15.0, 15.0)),
    )
    return (predict(fit.x), np.exp(fit.x))


def fit_exponentiated_weibull(times: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit scale/beta/gamma by log-time residuals and return predicted quantiles with the fitted parameters."""

    def predict(parameters: np.ndarray) -> np.ndarray:
        scale, beta, gamma = np.exp(parameters)
        return scale * (-np.log1p(-np.power(levels, 1.0 / gamma))) ** (1.0 / beta)

    fit = least_squares(
        lambda parameters: np.log(predict(parameters)) - np.log(times),
        np.log((np.median(times), 1.0, 1.0)),
        bounds=((-20.0, -4.0, -5.0), (20.0, 4.0, 7.0)),
        max_nfev=5000,
    )
    return (predict(fit.x), np.exp(fit.x))


def fit_rational_quadratic(times: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a monotone rational-quadratic quantile curve and return predictions with scale/control/weight."""
    unit = levels / levels[-1]

    def predict(parameters: np.ndarray) -> np.ndarray:
        scale = np.exp(parameters[0])
        control = 1.0 / (1.0 + np.exp(-parameters[1]))
        weight = np.exp(parameters[2])
        numerator = 2 * weight * (1 - unit) * unit * control + unit**2
        denominator = (1 - unit) ** 2 + 2 * weight * (1 - unit) * unit + unit**2
        return scale * numerator / denominator

    fit = least_squares(
        lambda parameters: np.log(predict(parameters)) - np.log(times),
        (np.log(times[-1]), -1.0, 0.0),
        bounds=((-20.0, -12.0, -10.0), (20.0, 12.0, 10.0)),
        max_nfev=3000,
    )
    parameters = np.asarray((np.exp(fit.x[0]), 1.0 / (1.0 + np.exp(-fit.x[1])), np.exp(fit.x[2])))
    return (predict(fit.x), parameters)


def survival_curve(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return a tail-normalized survival curve and its fixed affine parameters."""
    tail_count = min(501, max(51, values.size // 10))
    plateau = float(np.median(values[-tail_count:]))
    amplitude = float(values[0] - plateau)
    if amplitude <= 1e-14:
        return (np.full_like(values, np.nan), plateau, amplitude)
    return ((values - plateau) / amplitude, plateau, amplitude)
