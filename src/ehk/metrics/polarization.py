import dataclasses
from typing import Any, Optional

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import norm

from common.numerics.kde import (
    LINSPACE_SMPL_COUNT,
    fast_trapz,
    get_kde_pdf,
    js_divergence_continuous_fast,
    kl_divergence_continuous_fast,
)
from ehk.metrics.density import BINNED_POINTS, FittedDensity, evaluate_density
from ehk.metrics.state import MeasurementState


# Ideal-distribution helpers
def sample_linear_pdf(n: int, k: float, random_state=None):
    rng = np.random.default_rng(random_state)
    u = rng.uniform(0, 1, size=n)
    x = k - k * np.sqrt(1 - u)
    return x


def ideal_dist_init_array(x: np.ndarray, k: float = 2):
    ret_array = x / -k + 1
    ret_array[x > k] = 0
    ret_array[x < 0] = 0
    return ret_array


def ideal_dist_worst_obj_array(axis: np.ndarray, cl: float, effective_sample_size: int, min_bw: float):
    m = max(1, effective_sample_size)
    std = cl / 2
    factor = max(
        m ** (-1.0 / (1 + 4)),  # Scott's bandwidth rule.
        min_bw / std,
    )
    std_norm = std * factor

    o_worst_vals = 0.5 * norm.pdf(axis, 0, std_norm) + 0.5 * norm.pdf(axis, cl, std_norm)

    return o_worst_vals


def opinion_peak_count(opinions: NDArray, min_peak_distance: int = 50) -> int:
    """Count KDE modes on [-1, 1], including modes at the opinion boundaries."""
    grid = np.linspace(-1, 1, 1001)
    density = get_kde_pdf(opinions, 0.1, -1, 1)(grid)
    peaks, _ = find_peaks(np.pad(density, 1, mode="constant"), height=np.max(density) * 0.1, distance=min_peak_distance)
    return len(peaks)


# Opinion-distance result types
@dataclasses.dataclass
class DistanceResultDebugData:
    o_sample: NDArray
    s_sample: NDArray
    axis: NDArray
    rand_o_pdf: NDArray
    rand_s_pdf: NDArray
    worst_o_pdf: NDArray
    worst_s_pdf: NDArray


@dataclasses.dataclass
class DistanceResult:
    rand_o: float
    rand_s: float
    worst_o: float
    worst_s: float
    debug_data: Optional[DistanceResultDebugData] = None


@dataclasses.dataclass(frozen=True, eq=False)
class PreparedOpinion:
    owner: Any
    distances: NDArray
    density: FittedDensity | None
    objective_reference: NDArray
    objective_separation: float


class DistanceCalculator:
    def __init__(
        self,
        min_bandwidth: float = 0.05,
        sample_count: int = 500,
        max_kde_samples: int = 10000,
        t_err: float = 1e-10,
        k: float = 2.0,
        use_js_divergence: bool = False,
        use_debug_data: bool = False,
        density_method: str = "exact",
        bins: int = BINNED_POINTS,
    ):
        self.min_bandwidth = min_bandwidth
        self.sample_count = sample_count
        self.max_kde_samples = max_kde_samples

        self.t_err = t_err
        self.k = k

        self.use_js_divergence = use_js_divergence
        self.use_debug_data = use_debug_data
        self.density_method = density_method
        self.bins = bins
        self._evaluation_parameters(None, None)

        self.div = js_divergence_continuous_fast if use_js_divergence else kl_divergence_continuous_fast

        err_range = self.min_bandwidth * 4
        self.err_range = err_range
        self.axis = np.linspace(0 - err_range, k + err_range, LINSPACE_SMPL_COUNT)

        o_rand_smpl = sample_linear_pdf(int(sample_count**1.75), k, random_state=0)
        o_rand_pdf = get_kde_pdf(o_rand_smpl, self.min_bandwidth, 0, k)
        o_rand_vals = o_rand_pdf(self.axis)

        self.o_rand_vals = o_rand_vals
        self.triu_indices = np.triu_indices(sample_count, k=1)

        # Cache data-independent constants used by every collect() call.
        self._params_dict: Any = dict(xmin=0 - err_range, xmax=k + err_range, t_err=t_err, n=self.axis)
        s_worst_vals = norm.pdf(self.axis, 0, min_bandwidth)
        self.s_worst_vals = s_worst_vals
        if use_js_divergence:
            self._s_scale_rand = self._s_scale_worst = self.div(s_worst_vals, o_rand_vals, **self._params_dict)
        else:
            self._s_scale_rand = self.div(s_worst_vals, o_rand_vals, **self._params_dict)
            self._s_scale_worst = self.div(o_rand_vals, s_worst_vals, **self._params_dict)

    def _relative_js(self, observed, reference, clustered) -> float:
        def normalize(values):
            return values / fast_trapz(values, self.axis)

        observed, reference, clustered = map(normalize, (observed, reference, clustered))
        numerator = self.div(observed, reference, **self._params_dict)
        denominator = self.div(clustered, reference, **self._params_dict)
        # The historical numerical helper returns Jensen-Shannon distance.
        return float((numerator / denominator) ** 2)

    def _evaluation_parameters(self, method, bins):
        method = self.density_method if method is None else method
        bins = self.bins if bins is None else bins
        if method not in ("exact", "binned"):
            raise ValueError("density method must be 'exact' or 'binned'")
        if type(bins) is not int or bins < 256 or (bins - 1) % 255:
            raise ValueError("bins - 1 must be a positive multiple of the original 255 intervals")
        if method == "binned" and not self.use_js_divergence:
            raise ValueError("binned density evaluation is defined only for the JS indices")
        return method, bins

    def parameter_key(self):
        """Bind scalar caches to the actual calculator parameters and references."""
        return (
            self.min_bandwidth, self.sample_count, self.max_kde_samples, self.t_err, self.k,
            self.use_js_divergence, self.density_method, self.bins, id(self.div),
            self.axis.dtype.str, self.axis.shape, self.axis.tobytes(),
            self.o_rand_vals.dtype.str, self.o_rand_vals.shape, self.o_rand_vals.tobytes(),
            self.s_worst_vals.dtype.str, self.s_worst_vals.shape, self.s_worst_vals.tobytes(),
            self._s_scale_rand, self._s_scale_worst,
        )

    def _fit(self, samples):
        if not len(samples):
            return None
        values = np.array(samples, copy=True)
        values.setflags(write=False)
        pdf = get_kde_pdf(values, self.min_bandwidth, 0, self.k)
        constant = bool(np.all(values == values[0]))
        bandwidth = self.min_bandwidth if constant else float(pdf.cho_cov[0, 0])
        return FittedDensity(values, pdf, float(bandwidth), constant)

    def _subsample(self, distances, rng):
        if self.max_kde_samples > 0 and len(distances) > self.max_kde_samples:
            return distances[rng.choice(len(distances), self.max_kde_samples, replace=False)]
        return distances

    def _prepare_opinion(self, opinions, t_opinion, rng):
        i, j = self.triu_indices if len(opinions) == self.sample_count else np.triu_indices(len(opinions), k=1)
        distances = np.clip(np.abs(opinions[i] - opinions[j]), 0, self.k)
        sample = self._subsample(distances, rng)
        density = self._fit(sample)
        separated = distances[distances >= t_opinion]
        separation = float(t_opinion if separated.size == 0 else np.mean(separated))
        reference = ideal_dist_worst_obj_array(self.axis, separation, len(sample), self.min_bandwidth)
        reference.setflags(write=False)
        return PreparedOpinion(self, distances, density, reference, separation)

    def prepare_opinion(self, opinions, t_opinion: float = 0.4, *, random_state=None):
        """Prepare I_p using opinions alone, retaining the full-pair reference."""
        return self._prepare_opinion(np.asarray(opinions), t_opinion, np.random.default_rng(random_state))

    def _prepare_edge(self, state, rng):
        return self._fit(self._subsample(np.clip(state.edge_distances, 0, self.k), rng))

    def prepare_edge(self, state: MeasurementState, *, random_state=None):
        """Prepare I_s with the original RNG position after all-pair subsampling.

        Only the number of opinion pairs is needed to replay the first choice;
        graph-dependent measurements do not recompute all-pair differences.
        """
        rng = np.random.default_rng(random_state)
        pairs = len(state.opinions) * (len(state.opinions) - 1) // 2
        if self.max_kde_samples > 0 and pairs > self.max_kde_samples:
            rng.choice(pairs, self.max_kde_samples, replace=False)
        return self._prepare_edge(state, rng)

    def opinion_index(self, prepared: PreparedOpinion, *, method=None, bins=None, diagnostics=None) -> float:
        if prepared.owner is not self:
            raise ValueError("prepared opinion belongs to a different calculator")
        method, bins = self._evaluation_parameters(method, bins)
        values = evaluate_density(prepared.density, self.axis, method=method, bins=bins, diagnostics=diagnostics)
        if self.use_js_divergence:
            return self._relative_js(values, self.o_rand_vals, prepared.objective_reference)
        scale = self.div(prepared.objective_reference, self.o_rand_vals, **self._params_dict)
        return float(self.div(values, self.o_rand_vals, **self._params_dict) / scale)

    def edge_index(self, density: FittedDensity | None, *, method=None, bins=None, diagnostics=None) -> float:
        method, bins = self._evaluation_parameters(method, bins)
        values = evaluate_density(density, self.axis, method=method, bins=bins, diagnostics=diagnostics)
        if self.use_js_divergence:
            return self._relative_js(values, self.o_rand_vals, self.s_worst_vals)
        return float(self.div(values, self.o_rand_vals, **self._params_dict) / self._s_scale_rand)

    def calculate_indices(
        self, state: MeasurementState, t_opinion: float = 0.4, *, random_state=None, method=None, bins=None
    ) -> tuple[float, float]:
        """Assemble only I_p/I_s; omit archived worst-reference components."""
        self._evaluation_parameters(method, bins)
        rng = np.random.default_rng(random_state)
        opinion = self._prepare_opinion(state.opinions, t_opinion, rng)
        edge = self._prepare_edge(state, rng)
        return self.opinion_index(opinion, method=method, bins=bins), self.edge_index(edge, method=method, bins=bins)

    def calculate(
        self, digraph: nx.DiGraph, opinion: np.ndarray, t_opinion: float = 0.4, *, random_state=None,
        method=None, bins=None,
    ) -> DistanceResult:
        """Assemble graph inputs and all historical result components.

        Exact remains the default. The archived worst-reference fields always
        use exact PDFs, including when JS rand_o/rand_s request binned evaluation.
        Production callers needing only I_p/I_s use calculate_indices instead.
        """
        method, bins = self._evaluation_parameters(method, bins)
        state = MeasurementState.from_graph(digraph, opinion)
        rng = np.random.default_rng(random_state)
        objective = self._prepare_opinion(state.opinions, t_opinion, rng)
        edge = self._prepare_edge(state, rng)
        axis = self.axis
        s_rand_vals = o_rand_vals = self.o_rand_vals
        o_worst_vals, s_worst_vals = objective.objective_reference, self.s_worst_vals

        # The subjective scale is cached; the objective scale is data-dependent.
        params_dict = self._params_dict
        o_scale_worst = o_scale_rand = self.div(o_worst_vals, o_rand_vals, **params_dict)
        s_scale_rand = self._s_scale_rand
        s_scale_worst = self._s_scale_worst
        if not self.use_js_divergence:
            o_scale_worst = self.div(o_rand_vals, o_worst_vals, **params_dict)

        o_vals = evaluate_density(objective.density, axis, method="exact")
        s_vals = evaluate_density(edge, axis, method="exact")

        debug_data = None
        if self.use_debug_data:
            debug_data = DistanceResultDebugData(
                o_sample=objective.distances,
                s_sample=np.clip(state.edge_distances, 0, self.k),
                axis=axis,
                rand_o_pdf=o_rand_vals,
                rand_s_pdf=o_rand_vals,
                worst_o_pdf=o_worst_vals,
                worst_s_pdf=s_worst_vals,
            )

        if self.use_js_divergence:
            if method == "exact":
                rand_o = self._relative_js(o_vals, o_rand_vals, o_worst_vals)
                rand_s = self._relative_js(s_vals, s_rand_vals, s_worst_vals)
            else:
                rand_o = self.opinion_index(objective, method=method, bins=bins)
                rand_s = self.edge_index(edge, method=method, bins=bins)
        else:
            rand_o = self.div(o_vals, o_rand_vals, **params_dict) / o_scale_rand
            rand_s = self.div(s_vals, s_rand_vals, **params_dict) / s_scale_rand
        worst_o = self.div(o_vals, o_worst_vals, **params_dict) / o_scale_worst
        worst_s = self.div(s_vals, s_worst_vals, **params_dict) / s_scale_worst

        return DistanceResult(rand_o=rand_o, rand_s=rand_s, worst_o=worst_o, worst_s=worst_s, debug_data=debug_data)
