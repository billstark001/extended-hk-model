"""Typed scientific and numerical conventions for the paper's four indices."""

from dataclasses import asdict, dataclass, field, fields, replace

METRIC_REVISION = 6


@dataclass(frozen=True)
class DistanceKDEProtocol:
    evaluation: str = "linear_binned_gaussian"
    bins: int | None = 4081
    output_points: int = 256
    minimum_bandwidth: float = .01
    maximum_samples: int = 10000
    seed: str = "completed step"
    reference_evaluation: str = "exact"
    constant_samples: str = "analytic Gaussian"
    density_normalization: str = "unit trapezoidal mass on the finite 256-point integration grid"
    exact_audit: str = "DistanceCalculator density_method=exact; MeasurementEvaluator method=exact"


@dataclass(frozen=True)
class SamplingProtocol:
    absolute_midpoint_tolerance: float = .01
    maximum_interval: int = 512
    start: int = 0


@dataclass(frozen=True)
class HistoricalCutoff:
    value: float = .6
    status: str = "unvalidated descriptive low/high split"


@dataclass(frozen=True)
class MetricProtocol:
    revision: int = METRIC_REVISION
    I_p: str = "1 - JS-distance(observed all-pair distances, clustered) / JS-distance(clustered, random)"
    I_p_interpretation: str = "similarity to the equal-two-camp distance reference; consensus need not give zero; at most one"
    I_s: str = "1 - JS-distance(observed edge distances, clustered) / JS-distance(clustered, random)"
    clustered_objective: str = "equal two-camp reference; separation is mean observed distance >= confidence radius"
    distance_kde: DistanceKDEProtocol = field(default_factory=DistanceKDEProtocol)
    I_h: str = "max(0,(mean positive-outdegree agent edge concordance - b)/(1-b)); b=epsilon-epsilon**2/4"
    confidence: str = "absolute opinion distance strictly less than epsilon"
    I_w: str = "chronological signed trapezoids integral I_h dI_p + (1-I_p(T))*I_h(T)"
    endpoint_complement: bool = True
    endpoint_target: float | None = 1.
    bounds: str = "I_p and I_s are at most one and may be negative; retain signed I_w without clipping"
    sampling: SamplingProtocol = field(default_factory=SamplingProtocol)
    activity: str = "first I_s crossing max(0.98*terminal I_s,0.75); terminal step if absent"
    historical_iw_cutoff: HistoricalCutoff = field(default_factory=HistoricalCutoff)


METRIC_PROTOCOL = MetricProtocol()


def metric_protocol(*, endpoint_complement: bool = True) -> MetricProtocol:
    """Select the existing pathway convention without mutating the default."""
    if type(endpoint_complement) is not bool:
        raise TypeError("endpoint_complement must be boolean")
    if endpoint_complement:
        return METRIC_PROTOCOL
    return replace(
        METRIC_PROTOCOL,
        endpoint_complement=False,
        endpoint_target=None,
        I_w="chronological signed trapezoids integral I_h dI_p; endpoint complement explicitly disabled",
    )


@dataclass(frozen=True)
class RecordedDistanceKDEProtocol:
    evaluation: str | None = None
    bins: int | None = None
    output_points: int | None = None
    minimum_bandwidth: float | None = None
    maximum_samples: int | None = None
    seed: str | None = None
    reference_evaluation: str | None = None
    constant_samples: str | None = None
    density_normalization: str | None = None
    exact_audit: str | None = None
    _wire_fields: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __wire__(self, *, preserve_arrays=True):
        return {name: getattr(self, name) for name in self._wire_fields}


@dataclass(frozen=True)
class RecordedSamplingProtocol:
    absolute_midpoint_tolerance: float | None = None
    maximum_interval: int | None = None
    start: int | None = None
    _wire_fields: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __wire__(self, *, preserve_arrays=True):
        return {name: getattr(self, name) for name in self._wire_fields}


@dataclass(frozen=True)
class RecordedHistoricalCutoff:
    value: float | None = None
    status: str | None = None
    _wire_fields: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __wire__(self, *, preserve_arrays=True):
        return {name: getattr(self, name) for name in self._wire_fields}


@dataclass(frozen=True)
class RecordedMetricProtocol:
    """Partial archived evidence; absent conventions never acquire new defaults."""

    revision: int
    I_p: str | None = None
    I_p_interpretation: str | None = None
    I_s: str | None = None
    clustered_objective: str | None = None
    distance_kde: DistanceKDEProtocol | RecordedDistanceKDEProtocol | None = None
    I_h: str | None = None
    confidence: str | None = None
    I_w: str | None = None
    endpoint_complement: bool | None = None
    endpoint_target: float | None = None
    bounds: str | None = None
    sampling: SamplingProtocol | RecordedSamplingProtocol | None = None
    activity: str | None = None
    historical_iw_cutoff: HistoricalCutoff | RecordedHistoricalCutoff | None = None
    _wire_fields: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __wire__(self, *, preserve_arrays=True):
        result = {}
        for name in self._wire_fields:
            value = getattr(self, name)
            if hasattr(value, "__wire__"):
                result[name] = value.__wire__(preserve_arrays=preserve_arrays)
            elif hasattr(value, "__dataclass_fields__"):
                result[name] = asdict(value)
            else:
                result[name] = value
        return result


def parse_metric_protocol(value: dict) -> MetricProtocol | RecordedMetricProtocol:
    """Decode a file document, preserving every absent historical convention."""
    allowed = {item.name for item in fields(MetricProtocol)}
    if set(value) - allowed:
        raise ValueError(f"unknown metric protocol fields: {sorted(set(value) - allowed)}")
    if type(value.get("revision")) is not int:
        raise ValueError("recorded metric protocol requires an integer revision")
    decoded = dict(value)
    complete = set(decoded) == allowed
    for name, current, recorded in (
        ("distance_kde", DistanceKDEProtocol, RecordedDistanceKDEProtocol),
        ("sampling", SamplingProtocol, RecordedSamplingProtocol),
        ("historical_iw_cutoff", HistoricalCutoff, RecordedHistoricalCutoff),
    ):
        if name not in decoded:
            continue
        nested = decoded[name]
        if nested is None:
            complete = False
            continue
        nested_fields = {item.name for item in fields(current)}
        if not isinstance(nested, dict) or set(nested) - nested_fields:
            raise ValueError(f"invalid recorded metric protocol section: {name}")
        if set(nested) == nested_fields:
            decoded[name] = current(**nested)
        else:
            decoded[name] = recorded(**nested, _wire_fields=frozenset(nested))
            complete = False
    if complete:
        return MetricProtocol(**decoded)
    return RecordedMetricProtocol(**decoded, _wire_fields=frozenset(value))
