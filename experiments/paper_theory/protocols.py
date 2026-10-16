"""Frozen ensemble grids, replicate cohorts and analysis thresholds; horizons use tau=alpha*t."""

from itertools import product

from experiments.paper_theory.runtime import tasks as base

# Initial states are independent samples. Process replicates condition on each
# initial state; pooling both levels would inflate the effective sample size.
INITIAL_REPLICATES = (0, 1, 2, 3)
CLOCK_INITIAL_REPLICATES = (0, 1)
BASIN_PROCESS_REPLICATES = tuple(range(16))
CLOCK_PROCESS_REPLICATES = tuple(range(8))
POPULATIONS = (1000, 2000)
POPULATION_PROCESS_REPLICATES = tuple(range(4))
GRIDS = (81, 161, 241)
QUADRATURES = (9, 33)
ENSEMBLE_BINS = 161
ENSEMBLE_QUADRATURE = 33
MICRO_HORIZON = 30
CLOCK_HORIZON = 240
EXTENSION_HORIZON = 480
PROCESS_SEED_STRIDE = 1000000007
# Copy this protocol into the frozen plan; offline analysis must use that saved
# copy rather than thresholds edited after the run was generated.
ANALYSIS = {
    "bootstrap_samples": 4000,
    "bootstrap_seed": 2026100701,
    "confidence_level": 0.95,
    "tail_fractions": (0.1, 0.2),
    "clock_zero_tolerance": 1e-9,
    "snapshot_taus": (0, 0.1, 0.5, 1, 2, 4, 8, 16, 30, 60, 120, 240, 480),
    "extension_prefix_taus": (120, 240, 480),
}
# Original 14 cases failing the post hoc energy-only tail check, including all
# ten ratio=.125 main cases. Selection is fixed from the October 7 review.
EXTENSION_CASES = tuple((d, rec, z, 0.125, 0.45) for d, (rec, z) in product(base.DYNAMICS, base.RECS)) + (
    ("hk", "random", 1, 4, 0.45),
    ("hk", "opinion_random", 1, 16, 0.45),
    ("hk", "opinion_random", 4, 4, 0.45),
    ("hk", "opinion_random", 4, 0.25, 0.4),
)
CROSSING_RATIOS = {1: (1, 1.25, 1.5, 1.75, 2), 4: (0.5, 0.625, 0.75, 0.875, 1)}
HOLDOUT_FAILURES = (
    ("hk", "random", 1, 6, 60),
    ("deffuant", "structure_random_l0", 4, 6, 61),
    ("hk", "structure_random_l0", 4, 6, 61),
    ("deffuant", "structure_random_l0", 4, 0.5, 61),
)
