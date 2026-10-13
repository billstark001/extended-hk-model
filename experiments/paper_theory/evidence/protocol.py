"""Shared physical laws and analysis protocol; independent of design generators."""

PROTOCOL = {
    "schema": 1,
    "bootstrap_seed": 87231,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95,
    "plateau_window_tau": 20.0,
    "minimum_plateau_tau": 120.0,
    "plateau_relative_tolerance": 1e-4,
    "prefix_taus": [30.0, 120.0, 240.0, 480.0],
    "primary_clock": "restricted mean of normalized remaining energy in tau units",
    "primary_reference": "own long-window endpoint; fixed pair endpoint is a sensitivity control",
    "uncertainty_unit": "conditional processes and hierarchical initial-state/process bootstrap",
    "interpretation": "finite-window diagnostics; no infinite-mean or N-independent theorem",
}
KERNELS = (
    ("random", 1.0),
    ("opinion_random", 1.0),
    ("opinion_random", 4.0),
    ("structure_random_l0", 1.0),
    ("structure_random_l0", 4.0),
)
