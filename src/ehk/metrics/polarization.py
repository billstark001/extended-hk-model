"""Clustered-reference JS-distance indices with shared KDE preparation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import norm

from common.numerics.kde import LINSPACE_SMPL_COUNT, fast_trapz, get_kde_pdf, js_distance_continuous_fast
from ehk.metrics.density import BINNED_POINTS, FittedDensity, evaluate_density
from ehk.metrics.state import MeasurementState


def sample_random_distances(n: int, k: float, random_state=None):
    """Sample the triangular distance law of independent uniform opinions."""
    rng = np.random.default_rng(random_state)
    u = rng.uniform(0, 1, size=n)
    return k - k * np.sqrt(1 - u)


def clustered_objective_density(axis: np.ndarray, separation: float, sample_size: int, min_bw: float):
    """Smooth the distance law of two equally sized internally coherent camps."""
    std = separation / 2
    bandwidth = max(std * max(1, sample_size) ** (-1 / 5), min_bw)
    return .5 * norm.pdf(axis, 0, bandwidth) + .5 * norm.pdf(axis, separation, bandwidth)


def opinion_peak_count(opinions: NDArray, min_peak_distance: int = 50) -> int:
    """Count KDE modes on [-1, 1], including modes at the opinion boundaries."""
    grid = np.linspace(-1, 1, 1001)
    density = get_kde_pdf(opinions, 0.1, -1, 1)(grid)
    peaks, _ = find_peaks(np.pad(density, 1, mode="constant"), height=np.max(density) * 0.1, distance=min_peak_distance)
    return len(peaks)


@dataclass(frozen=True, eq=False)
class PreparedOpinion:
    owner: "DistanceCalculator"
    distances: NDArray
    density: FittedDensity | None
    clustered_reference: NDArray
    separation: float


class DistanceCalculator:
    """Evaluate 1 - d_JS(observed, clustered) / d_JS(clustered, random).

    Both indices use the same formula and numerical kernels. The opinion
    reference depends on all pair distances; the edge reference is fixed at
    zero distance. Explicit exact evaluation and alternate internal bin counts
    are retained for the existing numerical-accuracy study.
    """

    def __init__(
        self,
        min_bandwidth: float = .01,
        sample_count: int = 500,
        max_kde_samples: int = 10000,
        t_err: float = 1e-10,
        k: float = 2.0,
        density_method: str = "binned",
        bins: int = BINNED_POINTS,
    ):
        self.min_bandwidth = min_bandwidth
        self.sample_count = sample_count
        self.max_kde_samples = max_kde_samples
        self.t_err = t_err
        self.k = k
        self.density_method = density_method
        self.bins = bins
        self._evaluation_parameters(None, None)

        padding = min_bandwidth * 4
        self.axis = np.linspace(-padding, k + padding, LINSPACE_SMPL_COUNT)
        random_sample = sample_random_distances(int(sample_count**1.75), k, random_state=0)
        self.random_reference = get_kde_pdf(random_sample, min_bandwidth, 0, k)(self.axis)
        self.clustered_edge_reference = norm.pdf(self.axis, 0, min_bandwidth)
        self.triu_indices = np.triu_indices(sample_count, k=1)

    def _relative_js(self, observed, clustered) -> float:
        def normalize(values):
            return values / fast_trapz(values, self.axis)

        observed, clustered, random = map(normalize, (observed, clustered, self.random_reference))
        numerator = js_distance_continuous_fast(observed, clustered, n=self.axis, t_err=self.t_err)
        denominator = js_distance_continuous_fast(clustered, random, n=self.axis, t_err=self.t_err)
        return float(1 - numerator / denominator)

    def _evaluation_parameters(self, method, bins):
        method = self.density_method if method is None else method
        bins = self.bins if bins is None else bins
        if method not in ("exact", "binned"):
            raise ValueError("density method must be 'exact' or 'binned'")
        if type(bins) is not int or bins < 256 or (bins - 1) % 255:
            raise ValueError("bins - 1 must be a positive multiple of the original 255 intervals")
        return method, bins

    def parameter_key(self):
        """Bind scalar caches to the actual calculator parameters and references."""
        return (
            self.min_bandwidth, self.sample_count, self.max_kde_samples, self.t_err, self.k,
            self.density_method, self.bins,
            self.axis.dtype.str, self.axis.shape, self.axis.tobytes(),
            self.random_reference.dtype.str, self.random_reference.shape, self.random_reference.tobytes(),
            self.clustered_edge_reference.dtype.str, self.clustered_edge_reference.shape,
            self.clustered_edge_reference.tobytes(),
        )

    def _fit(self, samples):
        if not len(samples):
            return None
        values = np.array(samples, copy=True)
        values.setflags(write=False)
        pdf = get_kde_pdf(values, self.min_bandwidth, 0, self.k)
        constant = bool(np.all(values == values[0]))
        bandwidth = self.min_bandwidth if constant else float(pdf.cho_cov[0, 0])
        return FittedDensity(values, pdf, bandwidth, constant)

    def _subsample(self, distances, rng):
        if self.max_kde_samples > 0 and len(distances) > self.max_kde_samples:
            return distances[rng.choice(len(distances), self.max_kde_samples, replace=False)]
        return distances

    def _prepare_opinion(self, opinions, epsilon, rng):
        i, j = self.triu_indices if len(opinions) == self.sample_count else np.triu_indices(len(opinions), k=1)
        distances = np.clip(np.abs(opinions[i] - opinions[j]), 0, self.k)
        sample = self._subsample(distances, rng)
        density = self._fit(sample)
        separated = distances[distances >= epsilon]
        separation = float(epsilon if separated.size == 0 else np.mean(separated))
        reference = clustered_objective_density(self.axis, separation, len(sample), self.min_bandwidth)
        reference.setflags(write=False)
        return PreparedOpinion(self, distances, density, reference, separation)

    def prepare_opinion(self, opinions, epsilon: float = .4, *, random_state=None):
        """Prepare I_p using opinions alone and the full-pair clustered reference."""
        return self._prepare_opinion(np.asarray(opinions), epsilon, np.random.default_rng(random_state))

    def _prepare_edge(self, state, rng):
        return self._fit(self._subsample(np.clip(state.edge_distances, 0, self.k), rng))

    def prepare_edge(self, state: MeasurementState, *, random_state=None):
        """Replay the opinion subsample draw before sampling directed edges."""
        rng = np.random.default_rng(random_state)
        pairs = len(state.opinions) * (len(state.opinions) - 1) // 2
        if self.max_kde_samples > 0 and pairs > self.max_kde_samples:
            rng.choice(pairs, self.max_kde_samples, replace=False)
        return self._prepare_edge(state, rng)

    def opinion_index(
        self, prepared: PreparedOpinion, *, method=None, bins=None, diagnostics=None,
    ) -> float:
        if prepared.owner is not self:
            raise ValueError("prepared opinion belongs to a different calculator")
        method, bins = self._evaluation_parameters(method, bins)
        values = evaluate_density(prepared.density, self.axis, method=method, bins=bins, diagnostics=diagnostics)
        return self._relative_js(values, prepared.clustered_reference)

    def edge_index(
        self, density: FittedDensity | None, *, method=None, bins=None, diagnostics=None,
    ) -> float:
        method, bins = self._evaluation_parameters(method, bins)
        values = evaluate_density(density, self.axis, method=method, bins=bins, diagnostics=diagnostics)
        return self._relative_js(values, self.clustered_edge_reference)

    def calculate_indices(
        self, state: MeasurementState, epsilon: float = .4, *, random_state=None, method=None, bins=None
    ) -> tuple[float, float]:
        """Prepare and evaluate the sole I_p/I_s pair from one saved state."""
        self._evaluation_parameters(method, bins)
        rng = np.random.default_rng(random_state)
        opinion = self._prepare_opinion(state.opinions, epsilon, rng)
        edge = self._prepare_edge(state, rng)
        return self.opinion_index(opinion, method=method, bins=bins), self.edge_index(edge, method=method, bins=bins)
