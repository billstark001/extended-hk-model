"""Bounded memoization of the existing metrics at exactly equal saved states."""

from collections import OrderedDict

import numpy as np


class ExactMetricCache:
    """Reuse scalar results without changing sampling or retaining raw records.

    Absolute steps remain part of state identity because distance subsampling
    seeds its local RNG with that step. Graph edges and opinions retain their
    original order; neither rounded values nor sorted distributions are keys.
    Hits and misses count joint-result lookups, including the homophily lookup
    immediately following a distance lookup.
    """

    def __init__(self, max_entries=512, *, cross_records=True):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self.cross_records = cross_records
        self.hits = 0
        self.misses = 0
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def _key(self, record, step):
        step = int(step)
        if not self.cross_records:
            directory = getattr(record, "directory", None)
            return ("record", str(directory) if directory is not None else id(record), step)
        opinions = np.asarray(record.opinions[step])
        edges = np.asarray(list(record.get_graph(step).edges), dtype=np.int64).reshape(-1, 2)
        epsilon = float(record.metadata["HKParams"]["Tolerance"])
        return (
            "state", step, epsilon,
            opinions.dtype.str, opinions.shape, opinions.tobytes(order="C"),
            edges.dtype.str, edges.shape, edges.tobytes(order="C"),
        )

    def _values(self, record, step):
        key = self._key(record, step)
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        self.misses += 1
        # Read functions lazily: callers may install instrumentation before an
        # evaluation, without changing process-global metric implementations.
        from ehk.micro import observables

        ip, isub = observables.calc_distance_components(record, int(step))[:2]
        ih = observables.calc_homophily(record, int(step))
        values = (ip, isub, ih)
        self._entries[key] = values
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return values

    def distance(self, record, step):
        return self._values(record, step)[:2]

    def homophily(self, record, step):
        return self._values(record, step)[2]

    def indices(self, record):
        """Run the shared adaptive rule with explicitly supplied exact measurements."""
        from ehk.metrics.trajectory import sample_indices

        return sample_indices(
            record.max_step, lambda step: (*self.distance(record, step), self.homophily(record, step)),
        )
