"""Bounded scalar caches for the separate dependencies of I_p, I_s and I_h."""

import operator
from collections import OrderedDict

import numpy as np

from .state import MeasurementState


class _ScalarCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.values = OrderedDict()
        self.hits = self.misses = 0

    def resolve(self, key, compute):
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        self.misses += 1
        value = float(compute())
        self.values[key] = value
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value

    def clear(self):
        self.values.clear()
        self.hits = self.misses = 0

    def info(self):
        return {"hits": self.hits, "misses": self.misses, "size": len(self.values), "capacity": self.capacity}


class MeasurementEvaluator:
    """Evaluate saved states without retaining records, graphs or fitted KDEs.

    Opinion measurements depend on absolute step because it seeds subsampling.
    Edge measurements additionally depend on the ordered directed edges, while
    only the opinion reference depends on the confidence radius. Each
    edge preparation independently reproduces the preceding opinion RNG draw,
    so an opinion-cache hit cannot change the edge sample. Homophily has no
    sampling or KDE dependency and uses its own physical-state cache.

    ``point`` extracts one immutable MeasurementState for all three indices.
    Instances may be explicitly reused across records in one process; they do
    not install process-global state or synchronize concurrent callers.
    """

    def __init__(self, method="binned", bins=4081, max_entries=512, calculator=None):
        if method not in ("exact", "binned"):
            raise ValueError("method must be exact or binned")
        if type(bins) is not int or bins not in (1021, 2041, 4081, 8161):
            raise ValueError("bins must be one of 1021, 2041, 4081, 8161")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.method, self.bins = method, bins
        self._calculator = calculator
        self._opinion = _ScalarCache(max_entries)
        self._edge = _ScalarCache(max_entries)
        self._homophily = _ScalarCache(max_entries)

    @property
    def calculator(self):
        if self._calculator is None:
            from .polarization import DistanceCalculator

            self._calculator = DistanceCalculator()
        return self._calculator

    @staticmethod
    def _extract(record, step):
        if isinstance(step, (bool, np.bool_)):
            raise ValueError("step must be a nonnegative integer")
        try:
            step = operator.index(step)
        except TypeError as error:
            raise ValueError("step must be a nonnegative integer") from error
        if step < 0:
            raise ValueError("step must be a nonnegative integer")
        epsilon = float(record.metadata["HKParams"]["Tolerance"])
        state = MeasurementState.from_graph(record.get_graph(step), record.opinions[step])
        return state, int(step), epsilon

    @staticmethod
    def _physical_keys(state, epsilon):
        opinions = state.opinions
        edges = state.edges
        opinion = (epsilon, opinions.dtype.str, opinions.shape, opinions.tobytes(order="C"))
        edge = (edges.dtype.str, edges.shape, edges.tobytes(order="C"))
        return opinion, edge

    def _keys(self, state, step, epsilon):
        physical_opinion, ordered_edges = self._physical_keys(state, epsilon)
        parameters = (self.method, self.bins, self.calculator.parameter_key())
        opinion = (step, physical_opinion, parameters)
        edge = (step, physical_opinion[1:], ordered_edges, parameters)
        return opinion, edge, (physical_opinion, ordered_edges)

    def _distance(self, state, step, epsilon, keys):
        opinion_key, edge_key, _ = keys

        def opinion():
            prepared = self.calculator.prepare_opinion(state.opinions, epsilon, random_state=step)
            return self.calculator.opinion_index(prepared, method=self.method, bins=self.bins)

        def edge():
            prepared = self.calculator.prepare_edge(state, random_state=step)
            return self.calculator.edge_index(prepared, method=self.method, bins=self.bins)

        return self._opinion.resolve(opinion_key, opinion), self._edge.resolve(edge_key, edge)

    def _homophily_value(self, state, epsilon, key):
        def measure():
            from .homophily import mean_state_edge_concordance, normalize_homophily

            raw = mean_state_edge_concordance(state, epsilon)
            return normalize_homophily(raw, epsilon) if np.isfinite(raw) else float("nan")

        return self._homophily.resolve(key, measure)

    def point(self, record, step):
        """Return (I_p, I_s, I_h) after extracting opinions and topology once."""
        state, step, epsilon = self._extract(record, step)
        keys = self._keys(state, step, epsilon)
        ip, isub = self._distance(state, step, epsilon, keys)
        return ip, isub, self._homophily_value(state, epsilon, keys[2])

    def distance(self, record, step):
        """Return (I_p, I_s), reusing graph-independent opinion results."""
        state, step, epsilon = self._extract(record, step)
        return self._distance(state, step, epsilon, self._keys(state, step, epsilon))

    def homophily(self, record, step):
        """Compute only I_h, without fitting or evaluating either KDE."""
        state, _, epsilon = self._extract(record, step)
        return self._homophily_value(state, epsilon, self._physical_keys(state, epsilon))

    def indices(self, record):
        """Use the canonical adaptive sampler and full saved prefix."""
        from .trajectory import sample_indices

        return sample_indices(record.max_step, lambda step: self.point(record, int(step)))

    def clear(self):
        """Release all scalar entries and reset lookup counters."""
        for cache in (self._opinion, self._edge, self._homophily):
            cache.clear()

    def cache_info(self):
        return {"opinion": self._opinion.info(), "edge": self._edge.info(), "homophily": self._homophily.info()}
