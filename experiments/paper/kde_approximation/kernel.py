"""Approximate only observed KDE evaluation on the frozen 256-point metric grid.

Sampling, fitted bandwidths, references, Gaussian convolution and clustered-reference JS-distance similarity
evaluation belong to the metric package. This driver selects experimental B
values and records their diagnostics against explicit exact evaluation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehk.metrics.density import DensityDiagnostics, FittedDensity
from ehk.metrics.polarization import DistanceCalculator, PreparedOpinion
from ehk.metrics.state import MeasurementState

SUPPORTED_BINS = (1021, 2041, 4081, 8161)


def _readonly(values):
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _grid_stride(axis, bins):
    if type(bins) is not int or bins not in SUPPORTED_BINS:
        raise ValueError(f"bins must be one of {SUPPORTED_BINS}")
    if axis.shape != (256,) or not np.array_equal(axis, np.linspace(axis[0], axis[-1], len(axis))):
        raise ValueError("the original 256-point evenly spaced axis is required")
    return (bins - 1) // (len(axis) - 1)


@dataclass(frozen=True)
class PreparedDistance:
    """Worker-local fitted inputs shared by exact evaluation and every B candidate."""

    owner: "BinnedDistanceCalculator"
    step: int
    epsilon: float
    opinion: FittedDensity | None
    edge: FittedDensity | None
    objective_separation: float
    objective_reference: NDArray
    objective: PreparedOpinion


@dataclass(frozen=True)
class KernelDiagnostics:
    method: str
    bins: int | None
    stride: int | None
    opinion_samples: int
    edge_samples: int
    opinion_bandwidth: float | None
    edge_bandwidth: float | None
    opinion_constant: bool | None
    edge_constant: bool | None
    opinion_negative_count: int = 0
    edge_negative_count: int = 0
    opinion_minimum_before_clip: float | None = None
    edge_minimum_before_clip: float | None = None
    opinion_fft_length: int = 0
    edge_fft_length: int = 0


class BinnedDistanceCalculator:
    """Reuse the existing JS calculator, approximating only observed density evaluation."""

    def __init__(self, reference=None):
        if reference is None:
            from ehk.micro.observables import distance_collector

            reference = distance_collector
        if not isinstance(reference, DistanceCalculator):
            raise ValueError("an existing JS DistanceCalculator is required")
        _grid_stride(reference.axis, SUPPORTED_BINS[0])
        self.reference = reference
        self.axis = _readonly(reference.axis)
        self._random_reference = _readonly(reference.random_reference)
        self._subjective_reference = _readonly(reference.clustered_edge_reference)
        self._parameters = self._parameter_values()

    def _parameter_values(self):
        return tuple(getattr(self.reference, name) for name in (
            "min_bandwidth", "sample_count", "max_kde_samples", "t_err", "k",
        ))

    def _check_reference(self):
        if (
            self._parameter_values() != self._parameters
            or not np.array_equal(self.reference.axis, self.axis)
            or not np.array_equal(self.reference.random_reference, self._random_reference)
            or not np.array_equal(self.reference.clustered_edge_reference, self._subjective_reference)
        ):
            raise ValueError("the bound DistanceCalculator changed after preparation was configured")

    def prepare(self, graph, opinions, epsilon, step):
        """Fit the canonical components once for exact evaluation and every B."""
        self._check_reference()
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return self.prepare_state(MeasurementState.from_graph(graph, opinions), epsilon, step)

    def prepare_state(self, state, epsilon, step):
        self._check_reference()
        objective = self.reference.prepare_opinion(state.opinions, epsilon, random_state=step)
        edge = self.reference.prepare_edge(state, random_state=step)
        return PreparedDistance(
            self, step, float(epsilon), objective.density, edge,
            objective.separation, objective.clustered_reference, objective,
        )

    def _check_prepared(self, prepared):
        self._check_reference()
        if prepared.owner is not self:
            raise ValueError("prepared inputs belong to a different calculator")

    def _diagnostics(self, prepared, *, method, bins=None,
                     opinion_detail: DensityDiagnostics | None = None,
                     edge_detail: DensityDiagnostics | None = None):
        opinion, edge = prepared.opinion, prepared.edge
        opinion_detail = DensityDiagnostics() if opinion_detail is None else opinion_detail
        edge_detail = DensityDiagnostics() if edge_detail is None else edge_detail
        return KernelDiagnostics(
            method=method, bins=bins,
            stride=None if bins is None else _grid_stride(self.axis, bins),
            opinion_samples=0 if opinion is None else len(opinion.samples),
            edge_samples=0 if edge is None else len(edge.samples),
            opinion_bandwidth=None if opinion is None else opinion.bandwidth,
            edge_bandwidth=None if edge is None else edge.bandwidth,
            opinion_constant=None if opinion is None else opinion.constant,
            edge_constant=None if edge is None else edge.constant,
            opinion_negative_count=opinion_detail.negative_count,
            edge_negative_count=edge_detail.negative_count,
            opinion_minimum_before_clip=opinion_detail.minimum_before_clip,
            edge_minimum_before_clip=edge_detail.minimum_before_clip,
            opinion_fft_length=opinion_detail.fft_length,
            edge_fft_length=edge_detail.fft_length,
        )

    def exact(self, prepared):
        """Evaluate original fitted PDFs on the original axis; preparation cost is separate."""
        self._check_prepared(prepared)
        ip = self.reference.opinion_index(prepared.objective, method="exact")
        isub = self.reference.edge_index(prepared.edge, method="exact")
        return ip, isub, self._diagnostics(prepared, method="exact")

    def evaluate(self, prepared, bins):
        """Evaluate one linear-binning resolution, retaining exact constant-sample PDFs."""
        self._check_prepared(prepared)
        _grid_stride(self.axis, bins)
        opinion_detail, edge_detail = DensityDiagnostics(), DensityDiagnostics()
        ip = self.reference.opinion_index(prepared.objective, method="binned", bins=bins, diagnostics=opinion_detail)
        isub = self.reference.edge_index(prepared.edge, method="binned", bins=bins, diagnostics=edge_detail)
        diagnostics = self._diagnostics(prepared, method="linear_binned_gaussian", bins=bins,
                                        opinion_detail=opinion_detail, edge_detail=edge_detail)
        return ip, isub, diagnostics
