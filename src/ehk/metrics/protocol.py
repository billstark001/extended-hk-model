"""Declared conventions for the paper's existing four indices."""

from copy import deepcopy

METRIC_REVISION = 6
METRIC_PROTOCOL = {
    "revision": METRIC_REVISION,
    "I_p": "1 - JS-distance(observed all-pair distances, clustered) / JS-distance(clustered, random)",
    "I_p_interpretation": "similarity to the equal-two-camp distance reference; consensus need not give zero; at most one",
    "I_s": "1 - JS-distance(observed edge distances, clustered) / JS-distance(clustered, random)",
    "clustered_objective": "equal two-camp reference; separation is mean observed distance >= confidence radius",
    "distance_kde": {
        "evaluation": "linear_binned_gaussian", "bins": 4081, "output_points": 256,
        "minimum_bandwidth": 0.01, "maximum_samples": 10000, "seed": "completed step",
        "reference_evaluation": "exact", "constant_samples": "analytic Gaussian",
        "density_normalization": "unit trapezoidal mass on the finite 256-point integration grid",
        "exact_audit": "DistanceCalculator density_method=exact; MeasurementEvaluator method=exact",
    },
    "I_h": "max(0,(mean positive-outdegree agent edge concordance - b)/(1-b)); b=epsilon-epsilon**2/4",
    "confidence": "absolute opinion distance strictly less than epsilon",
    "I_w": "chronological signed trapezoids integral I_h dI_p + (1-I_p(T))*I_h(T)",
    "endpoint_complement": True,
    "endpoint_target": 1.0,
    "bounds": "I_p and I_s are at most one and may be negative; retain signed I_w without clipping",
    "sampling": {"absolute_midpoint_tolerance": 0.01, "maximum_interval": 512, "start": 0},
    "activity": "first I_s crossing max(0.98*terminal I_s,0.75); terminal step if absent",
    "historical_iw_cutoff": {"value": 0.6, "status": "unvalidated descriptive low/high split"},
}


def metric_protocol(*, endpoint_complement=True):
    """Return a separate protocol describing the selected pathway convention."""
    if type(endpoint_complement) is not bool:
        raise TypeError("endpoint_complement must be boolean")
    protocol = deepcopy(METRIC_PROTOCOL)
    protocol["endpoint_complement"] = endpoint_complement
    protocol["endpoint_target"] = 1.0 if endpoint_complement else None
    if not endpoint_complement:
        protocol["I_w"] = "chronological signed trapezoids integral I_h dI_p; endpoint complement explicitly disabled"
    return protocol
