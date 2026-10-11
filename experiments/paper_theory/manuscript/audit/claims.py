"""Check historical manuscript claims against frozen figure/review/run evidence."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from ...paths import MANUSCRIPT_INPUT, REVIEW_INPUT, REVISION_INPUT
from ..paths import FIGURES_OUTPUT

HERE = FIGURES_OUTPUT
EXPERIMENT = REVIEW_INPUT.resolve()
CORE = EXPERIMENT / "review_20261007"
FOLLOWUP = EXPERIMENT / "review_followup_20261007"
records = []
sources = {}


def read(path):
    """Return CSV rows and register the complete source-file SHA256 in the audit provenance map."""
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))


def read_json(path):
    """Read JSON and register the complete source-file SHA256 in the audit provenance map."""
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def check(name, observed, expected, tolerance=0, scope=""):
    """Assert a scalar or sequence claim within absolute tolerance, then append its observed/expected values and scope to the audit."""
    if isinstance(observed, (list, tuple)):
        ok = len(observed) == len(expected) and all(abs(a - b) <= tolerance for a, b in zip(observed, expected))
    else:
        ok = abs(observed - expected) <= tolerance
    if not ok:
        raise AssertionError(name)
    records.append(dict(claim=name, observed=observed, expected=expected, absolute_tolerance=tolerance, scope=scope))


def main():
    """Audit the fixed manuscript claims against complete recorded datasets and write check/provenance outputs."""
    global HERE, EXPERIMENT, CORE, FOLLOWUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure-input", type=Path)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    HERE = args.output.expanduser().resolve() if args.output else FIGURES_OUTPUT
    EXPERIMENT = args.review_root.expanduser().resolve() if args.review_root else REVIEW_INPUT.resolve()
    CORE, FOLLOWUP = (EXPERIMENT / "review_20261007", EXPERIMENT / "review_followup_20261007")
    run_root = args.run_root.expanduser().resolve() if args.run_root else REVISION_INPUT.resolve()
    figure_input = args.figure_input.expanduser().resolve() if args.figure_input else MANUSCRIPT_INPUT.resolve()
    records.clear()
    sources.clear()
    HERE.mkdir(parents=True, exist_ok=True)
    clocks = read(FOLLOWUP / "conditional_clock_prediction.csv")
    check(
        "conditional clock cases",
        len(clocks),
        24,
        scope="two dynamics, three configurations, two initial graphs, two energies",
    )
    for channel, expected in (("node", (2.88, 18.27)), ("edge", (0.12, 4.36))):
        es = [100 * float(r["relative_error"]) for r in clocks if r["channel"] == channel]
        check(
            channel + " clock error percent range",
            [min(es), max(es)],
            expected,
            0.006,
            "conditional clocks of eight-process mean curves; tau=30",
        )
        check(
            channel + " pair predictions inside mean CI",
            sum(r["pair_within_mean_interval"] == "True" for r in clocks if r["channel"] == channel),
            1 if channel == "node" else 5,
            scope="pointwise 95% conditional mean intervals",
        )
    contrasts = read(FOLLOWUP / "recommendation_contrasts.csv")
    for quantity, expected in (
        ("node", (3.672840, 3.678025, 3.526951, 3.414835)),
        ("edge", (1.185970, 1.135514, 1.193366, 1.133033)),
    ):
        index = {(r["dynamics"], int(r["initial"])): r for r in contrasts if r["quantity"] == quantity}
        observed = [float(index[k]["micro_factor"]) for k in (("hk", 0), ("hk", 1), ("deffuant", 0), ("deffuant", 1))]
        check(
            quantity + " Opinion4/Random factors",
            observed,
            expected,
            1e-6,
            "secondary paired process bootstrap; fixed initial graph",
        )
    check("all contrast bootstrap clocks valid", min(float(r["valid_bootstrap_fraction"]) for r in contrasts), 1)
    check(
        "differential contrast CI lower bound above 2.8",
        min(float(r["lower"]) for r in contrasts if r["quantity"] == "differential") > 2.8,
        True,
    )

    states = read(FOLLOWUP / "state_energy_separation.csv")
    check("new mesoscopic clock trajectories", len(states), 110)
    check("energy confirmations", sum(r["energy_confirmed"] == "True" for r in states), 109)
    check("joint confirmations", sum(r["joint_confirmed"] == "True" for r in states), 33)
    extensions = read(FOLLOWUP / "extended_platforms.csv")
    check("extended previous unresolved cases", len(extensions), 14)
    check("extended energy confirmations", sum(r["energy_confirmed"] == "True" for r in extensions), 14)
    check("extended joint confirmations", sum(r["joint_confirmed"] == "True" for r in extensions), 9)
    late = next(
        r
        for r in extensions
        if r["dynamics"] == "hk"
        and r["recommender"] == "opinion_random"
        and float(r["zeta"]) == 4
        and float(r["ratio"]) == 4
    )
    check(
        "late node scaled half-time", float(late["rho_tau50"]), 93.65, 0.005, "mesoscopic B161/Q9 only; not microscopic"
    )
    check("late edge scaled half-time", float(late["edge_tau50"]), 1.334, 0.0005, "same mesoscopic case")
    check("late joint stopping tau", float(late["actual_tau"]), 160)
    check("late node tail sensitivity percent", 100 * float(late["rho_tail_relative_change"]), 0.123, 0.0005)
    direct = read(run_root / "revision_followup_v1/analysis/clocks/direct_clocks.csv")
    check(
        "progress-order-inconsistent refined trajectories",
        sum(r["order_consistent"] == "False" for r in direct if r["role"] == "crossing"),
        36,
    )
    refined = [r for r in direct if r["role"] == "crossing"]
    check("refined trajectory count", len(refined), 80)
    check(
        "refined nonmonotone energy flags",
        sum(r[k] == "True" for r in refined for k in ("rho_nonmonotone_release", "edge_nonmonotone_release")),
        0,
    )

    ranges = read(FOLLOWUP / "crossing_ranges.csv")
    index = {(r["dynamics"], float(r["zeta"]), float(r["progress"])): r for r in ranges}
    for k, expected in {
        ("hk", 1, 50): (1.3233, 1.3276),
        ("deffuant", 1, 50): (1.3212, 1.3238),
        ("hk", 4, 50): (0.8660, 0.8698),
        ("deffuant", 4, 50): (0.8628, 0.8654),
    }.items():
        r = index[k]
        check(
            "crossing range " + str(k),
            [float(r["interpolation_min"]), float(r["interpolation_max"])],
            expected,
            0.00005,
            "four numerical settings, not CI",
        )
    check("HK Opinion4 T90 numerical configurations", int(index[("hk", 4, 90)]["numeric_configurations"]), 3)
    grid = read(FOLLOWUP / "conditional_ensemble_grid.csv")
    index = {(int(r["initial"]), int(r["pair_bins"]), int(r["quadrature"])): r for r in grid}
    for k, expected in {
        (2, 81, 33): 0.328,
        (2, 241, 33): 0.030,
        (3, 81, 33): 0.370,
        (3, 241, 33): 0.223,
        (0, 241, 33): 0.201,
    }.items():
        check(
            "conditional-mean W1 " + str(k),
            float(index[k]["mean_rho_w1"]),
            expected,
            0.0005,
            "same 16 processes; common 81-bin projection; tau=30",
        )
    pop = read(FOLLOWUP / "population_aggregate.csv")
    check("population aggregate sizes", [int(float(r["n"])) for r in pop], [500, 1000, 2000])
    check("population median individual W1", [float(r["node_w1_median"]) for r in pop], [0.283, 0.0156, 0.0165], 0.0005)
    check(
        "population median individual node gap percent",
        [100 * float(r["node_energy_gap_median"]) for r in pop],
        [26.09, 1.16, 9.81],
        0.005,
    )

    equal = read(FOLLOWUP / "population_equal_repeats.csv")
    match = next(r for r in equal if int(float(r["n"])) == 1000 and int(float(r["initial_seed_index"])) == 0)
    check(
        "N1000 initial0 mean-law W1", float(match["mean_rho_w1"]), 0.0104, 0.00005, "four processes; B161 Q33; tau=30"
    )
    check("N1000 initial0 micro scaled clock", float(match["node_micro_tau50"]), 8.19, 0.005)
    check("N1000 initial0 pair scaled clock", float(match["node_pair_tau50"]), 24.01, 0.005)
    continuous = read(CORE / "micro_continuous_energy.csv")
    for ratio, expected in ((0.25, (0.550, 1.941)), (2, (0.642, 0.738)), (10, (1.905, 0.978))):
        selected = [r for r in continuous if r["tier"] == "core" and float(r["ratio"]) == ratio]
        check("unscreened core cases r=" + str(ratio), len(selected), 40)
        check(
            "continuous micro energy median percent r=" + str(ratio),
            [100 * statistics.median(float(r[k]) for r in selected) for k in ("exact_node_gap", "exact_edge_gap")],
            expected,
            0.0006,
        )
    check(
        "core r10 maximum continuous node gap percent",
        100 * max(float(r["exact_node_gap"]) for r in continuous if r["tier"] == "core" and float(r["ratio"]) == 10),
        43.5,
        0.05,
    )

    core_clocks = read(CORE / "clocks_reviewed.csv")
    check("core budget positive motion cases", sum(float(r["budget_motion"]) > 1e-12 for r in core_clocks), 11)
    check("core positive cumulative rewiring cases", sum(float(r["budget_rewire"]) > 1e-12 for r in core_clocks), 0)
    check("core max budget residual", max(abs(float(r["budget_residual"])) for r in core_clocks), 2.39e-15, 0.01e-15)
    check(
        "core maximum node mass residual", max(abs(float(r["mass_residual"])) for r in core_clocks), 2.58e-13, 0.01e-13
    )
    check("core maximum row residual", max(abs(float(r["row_residual"])) for r in core_clocks), 9.47e-12, 0.01e-12)
    check("core original joint confirmations", sum(r["primary_joint_plateau"] == "True" for r in core_clocks), 132)
    check("core post-hoc energy-only additions", sum(r["energy_only_added"] == "True" for r in core_clocks), 56)
    check("core energy-unconfirmed cases", sum(r["energy_only_final_pass"] == "False" for r in core_clocks), 14)
    overlaps = read(FOLLOWUP / "baseline_overlap_checks.csv")
    check("overlap microscopic trajectories", sum(r["family"] == "micro" for r in overlaps), 8)
    check("overlap mesoscopic scalar prefixes", sum(r["family"] == "clocks" for r in overlaps), 30)
    check(
        "overlap max scalar difference",
        max(float(r["max_scalar_gap"]) for r in overlaps),
        3.2612801348363973e-16,
        1e-20,
    )
    summary = read_json(CORE / "summary.json")
    check("iid list mean audit max absolute error", summary["kernel_max_mean_error"], 7.30e-15, 0.01e-15)
    check("iid HK second moment max absolute error", summary["kernel_max_second_error"], 4.27e-15, 0.01e-15)
    check("same-list MC maximum standard errors", summary["kernel_max_gap_mc_se_units"], 1.72, 0.005)
    for q, expected in ((9, (0.00391, 0.0586)), (33, (0.00255, 0.0168))):
        r = next(r for r in summary["kernel_groups"] if int(r["degree"]) == 15 and int(r["quadrature"]) == q)
        check(
            "HK normal/exact TV Q=" + str(q),
            [r["normal_exact_tv_median"], r["normal_exact_tv_max"]],
            expected,
            0.00002,
            "binned iid center-confidence destination law",
        )
    for alpha, expected in (
        (0.01, (0.0137, 0.723, 0.143, 0.140)),
        (0.05, (0.0140, 0.778, 0.248, 0.288)),
        (0.2, (0.0187, 0.818, 0.582, 0.726)),
    ):
        r = next(r for r in summary["operator_by_alpha"] if float(r["alpha"]) == alpha)
        check("operator cases alpha=" + str(alpha), r["cases"], 30)
        check(
            "operator W1 and Hellinger alpha=" + str(alpha),
            [r["node_w1_median"], r["node_hellinger_median"]],
            expected[:2],
            0.0005,
            "epsilon .4, tau20, diagonal local versus nonlocal",
        )
        check(
            "operator energy-gap percent alpha=" + str(alpha),
            [100 * r["node_energy_gap_median"], 100 * r["edge_energy_gap_median"]],
            expected[2:],
            0.0006,
        )
    holdout = read(FOLLOWUP / "holdout_grid.csv")
    for key, expected in {
        ("hk", "random", 6): (0.339, 40.4),
        ("deffuant", "structure_random_l0", 6): (0.370, 18.2),
        ("hk", "structure_random_l0", 6): (0.346, 24.9),
        ("deffuant", "structure_random_l0", 0.5): (0.361, 55.5),
    }.items():
        r = next(
            r
            for r in holdout
            if int(r["b"]) == 241
            and int(r["quadrature"]) == 33
            and (r["dynamics"], r["recommender"], float(r["ratio"])) == key
        )
        check(
            "finer-grid held-out W1 " + str(key),
            float(r["node_w1"]),
            expected[0],
            0.0005,
            "selected single-process stress case, no ensemble interval",
        )
        check("finer-grid held-out node gap percent " + str(key), 100 * float(r["node_energy_gap"]), expected[1], 0.05)
    budget = read_json(figure_input / "rewiring_sign_audit.json")
    check("raw discrete-budget audited trajectories", len(budget["records"]), 202)
    check(
        "positive saved-interval rewiring increments",
        budget["positive_interval_count"],
        0,
        scope="saved intervals, not every unsaved physical substep",
    )
    landscape = read_json(figure_input / "provenance.json")["figure_cases"]["energy_budget_landscape"]
    check(
        "illustrated rewiring share of edge loss percent",
        100 * landscape["rewiring_fraction_of_edge_loss_at_tau24"],
        84.9,
        0.05,
        "HK Opinion4 r2, analytic uniform state, tau24, finite-step decomposition",
    )
    check(
        "occupied uphill drift rows",
        landscape["uphill_rows_at_snapshot"] > 0,
        True,
        scope="HK Opinion4 r2 at tau4; scaled drift times energy slope above 1e-8",
    )
    display = read(figure_input / "tables/mesoscopic_clock_display.csv")
    check("main clock scan parameter cases", len(display), 100)
    check(
        "extended cases in main epsilon .45 scan",
        sum(r["confirmation_source"] == "new frozen extension protocol" for r in display),
        13,
    )

    out = {
        "purpose": "manuscript numerical-claim audit; original simulations unchanged",
        "checks": records,
        "input_table_sha256": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "claim_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print("numeric claims checked", len(records))


if __name__ == "__main__":
    main()
