"""Parameters for the matched measure model."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    """Immutable physical and numerical settings for measure sweeps.

    alpha is influence, ratio is q/alpha, and dt is physical time per update.
    horizon and record_tau use scaled time tau=alpha*t; bins is the cell count
    for node/edge mass arrays.
    Call validate before constructing a runtime request.
    """

    dynamics: str = "hk"
    recommender: str = "random"
    epsilon: float = 0.4
    alpha: float = 0.01
    ratio: float = 2.0
    degree: int = 15
    recommendations: int = 10
    bins: int = 81
    population: int = 500
    dt: float = 1.0
    horizon: float = 30.0
    record_tau: float = 0.05
    steepness: float = 1.0
    opinion_tolerance: float = 0.4
    quadrature: int = 9

    @property
    def q(self):
        """Return the physical rewiring probability q = alpha * ratio."""
        return self.alpha * self.ratio

    def validate(self):
        """Reject unsupported closures, invalid sizes and per-step update probabilities exceeding one."""
        if self.dynamics not in ("hk", "deffuant"):
            raise ValueError("unknown opinion rule")
        if self.recommender not in ("random", "opinion_random", "structure_random_l0"):
            raise ValueError("use an N-independent pair closure recommender")
        for name in ("degree", "recommendations", "bins", "population", "quadrature"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.bins < 2 or self.population <= max(self.degree, self.recommendations):
            raise ValueError("invalid population/grid size")
        values = [
            self.epsilon,
            self.alpha,
            self.ratio,
            self.dt,
            self.horizon,
            self.record_tau,
            self.steepness,
            self.opinion_tolerance,
        ]
        if not np.isfinite(values).all() or not 0 < self.epsilon <= 2:
            raise ValueError("parameters must be finite; epsilon must lie in (0,2]")
        if (
            min(self.alpha, self.dt, self.horizon, self.record_tau, self.steepness, self.opinion_tolerance) <= 0
            or self.ratio < 0
        ):
            raise ValueError("positive update/time scales and nonnegative ratio required")
        if self.alpha > 1 or self.q > 1 or self.dt * max(self.alpha, self.q) > 1:
            raise ValueError("alpha, q and their per-step probabilities must not exceed one")
