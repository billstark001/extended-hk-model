"""Declared conventions for the paper's existing four indices."""

METRIC_REVISION = 4
METRIC_PROTOCOL = {
    "revision": METRIC_REVISION,
    "I_p": "JS-divergence(observed all-pair distances, random) / JS-divergence(clustered, random)",
    "I_p_interpretation": "deviation from random distance distribution; may rise under consensus; not bounded by one",
    "I_s": "JS-divergence(observed edge distances, random) / JS-divergence(clustered, random)",
    "clustered_objective": "equal two-camp reference; separation is mean observed distance >= confidence radius",
    "distance_kde": {
        "evaluation": "linear_binned_gaussian", "bins": 4081, "output_points": 256,
        "minimum_bandwidth": 0.01, "maximum_samples": 10000, "seed": "completed step",
        "reference_evaluation": "exact", "constant_samples": "exact Gaussian fallback",
        "exact_audit": "DistanceCalculator density_method=exact; MeasurementEvaluator method=exact",
    },
    "I_h": "max(0,(mean positive-outdegree agent edge concordance - b)/(1-b)); b=epsilon-epsilon**2/4",
    "confidence": "absolute opinion distance strictly less than epsilon",
    "I_w": "chronological signed trapezoids integral I_h dI_p; no endpoint complement",
    "bounds": "retain unbounded I_p and signed I_w without clipping",
    "sampling": {"absolute_midpoint_tolerance": 0.01, "maximum_interval": 512, "start": 0},
    "activity": "first I_s crossing max(0.98*terminal I_s,0.75); terminal step if absent",
    "historical_iw_cutoff": {"value": 0.6, "status": "unvalidated descriptive low/high split"},
}
