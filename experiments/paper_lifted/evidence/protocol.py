"""Fixed science; cost calibration changes replication, never physical rules."""

from pathlib import Path

from common.io.json import sha

MODEL_ROOT = Path(__file__).resolve().parents[3]
RATIOS = (0.25, 2.0, 10.0)
RATIO_GRID = (0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 10, 16, 32)
OBSERVATION_TAUS = (0, 1, 4, 16, 30, 120)
PROTOCOL = {
    "dynamics": "hk",
    "recommender": "random",
    "epsilon": 0.4,
    "alpha": 0.02,
    "dt": 1,
    "degree": 15,
    "recommendations": 10,
    "bins": 161,
    "quadrature": 33,
    "repost_rate": 0,
    "post_retain_count": 1,
    "noise_diffusion": 0,
    "network": "fixed_out_degree",
    "initial": "iid_uniform[-1,1]",
    "outcome": "all-agent confidence components; no occupied/major mass discarded",
    "deltas": [0.1, 0.05],
    "confidence": 0.95,
    "terminal_horizon_tau": 480,
    "extension_horizon_tau": 960,
    "observation_taus": list(OBSERVATION_TAUS),
    "probability_bounds": "Clopper-Pearson/Bonferroni over declared state-class family, including unresolved mass",
    "time_object": "first hit on observation grid; separate estimate/possible/confirmed times",
    "independence": "initial-state groups; independent mother, screen, confirmation and refinement streams",
    "finite_deadline": "horizon is relative to each query; eventual probabilities bracket unknown tails",
    "refinement": "independent validation samples; selected cases never estimate population prevalence",
    "followup_query_reserve": 42,
    "moments": "joint cell-mass increments, covariance factors, no dense covariance",
    "partial": "failed/deadline-deferred tasks are missing work, not terminal outcomes",
    "measure_reference": "current theory cell_average measure map; lifted has bin-center confidence",
}


def sources(root=MODEL_ROOT):
    """Scientific Python/adapter source binding, excluding docs, tests and generated data."""
    paths: list[Path] = []
    for directory in (root / "src/lifted", root / "experiments/paper_lifted"):
        paths.extend(
            p
            for p in directory.rglob("*")
            if (p.suffix in (".py", ".go") or p.name == "go.mod") and not p.name.endswith("_test.go")
        )
    paths.extend(
        root / p
        for p in (
            "src/theory/parameters.py",
            "src/theory/runtime/population.py",
            "src/theory/runtime/measure.py",
            "src/theory/runtime/protocol.py",
        )
    )
    for directory in (root / "src/common/experiments", root / "src/common/runtime"):
        paths.extend(directory.glob("*.py"))
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}
