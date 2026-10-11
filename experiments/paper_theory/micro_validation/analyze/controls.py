"""Manuscript population contrasts with independent initials as statistical units."""

from __future__ import annotations

import collections

import numpy as np

from common.io.tables import table
from common.numerics.distributions import rebin_matrix
from common.statistics import bootstrap_weights
from experiments.paper_theory.manuscript.reports.tables import number
from experiments.paper_theory.manuscript.reports.tables import read_table as read_csv
from theory.release import batch_half_times
from theory.validation import trajectory_gap


def ensemble_grid(micro, protocol, out):
    """Reuse mesh-invariant micro paths; W1 always uses common 81-bin masses.

    Conservative rebinning permits comparison with fine pair predictions
    without inventing sub-bin microscopic opinions. Energy paths are the
    recorded continuous micro energies and need no rebinning.
    """
    answer = []
    weights = bootstrap_weights(16, protocol)
    for initial in range(4):
        rs = sorted(
            [
                r
                for r in micro
                if r["meta"]["n"] == 500
                and r["meta"]["ratio"] == 10
                and r["meta"]["alpha"] == 0.02
                and r["meta"]["b"] == 161
                and r["meta"]["quadrature"] == 33
                and r["meta"]["initial_replicate"] == initial
            ],
            key=lambda r: r["meta"]["process_replicate"],
        )
        assert len(rs) == 16 and len({r["initial_hash"] for r in rs}) == 1
        times = rs[0]["times"]
        mass = np.asarray([r["rho"] for r in rs]) @ rebin_matrix(161, 81).T
        boot_mass = weights @ mass
        mean_mass = mass.mean(axis=0)
        node = np.asarray([r["node"] for r in rs])
        boot_node = weights @ node
        for pair in [
            r for r in micro if r["meta"]["role"] == "basin_grid" and r["meta"]["initial_replicate"] == initial
        ]:
            m = pair["meta"]
            assert pair["initial_hash"] == rs[0]["initial_hash"]
            assert np.array_equal(pair["times"], times)
            assert np.max(np.abs(pair["node"] - rs[0]["node"])) < 1e-12
            reference = np.asarray(pair["pair_rho"])
            if m["b"] != 81:
                reference = rebin_matrix(m["b"], 81) @ reference
            reference /= reference.sum()
            point = np.abs(np.cumsum(mean_mass / mean_mass.sum() - reference)).sum() * 2 / 81
            boot_w1 = (
                np.abs(np.cumsum(boot_mass / boot_mass.sum(axis=1)[:, None] - reference, axis=1)).sum(axis=1) * 2 / 81
            )
            low, high = np.quantile(boot_w1, (0.025, 0.975))
            scale = (times[-1] - times[0]) * node[0, 0]
            g = np.trapezoid(np.abs(node.mean(axis=0) - pair["node_pair"]), times) / scale
            bg = np.trapezoid(np.abs(boot_node - pair["node_pair"]), times, axis=1) / scale
            gl, gh = np.quantile(bg, (0.025, 0.975))
            answer.append(
                {
                    "initial": initial,
                    "pair_bins": m["b"],
                    "quadrature": m["quadrature"],
                    "common_comparison_bins": 81,
                    "processes": 16,
                    "pair_id": pair["id"],
                    "mean_rho_w1": float(point),
                    "w1_lower": float(low),
                    "w1_upper": float(high),
                    "node_gap_of_mean_curve": float(g),
                    "node_gap_lower": float(gl),
                    "node_gap_upper": float(gh),
                    "scope": "secondary reuse; fixed initial graph; conservative common-bin projection",
                }
            )
    table(out / "conditional_ensemble_grid.csv", answer)
    return answer


def mechanism_contrasts(micro, protocol, out):
    """Secondary paired bootstrap of the preselected Opinion4/Random contrast."""
    answer = []
    weights = bootstrap_weights(8, protocol)
    tail = (1 - protocol["confidence_level"]) / 2
    for d in ("hk", "deffuant"):
        for initial in (0, 1):
            samples = {
                rec: sorted(
                    [
                        r
                        for r in micro
                        if r["meta"]["role"] == "clock_ensemble"
                        and r["meta"]["dynamics"] == d
                        and r["meta"]["initial_replicate"] == initial
                        and r["meta"]["recommender"] == rec
                    ],
                    key=lambda r: r["meta"]["process_replicate"],
                )
                for rec in ("random", "opinion_random")
            }
            assert all(len(s) == 8 for s in samples.values())
            assert [r["meta"]["process_seed"] for r in samples["random"]] == [
                r["meta"]["process_seed"] for r in samples["opinion_random"]
            ]
            values, boots, pairs = {}, {}, {}
            for channel in ("node", "edge"):
                half, resampled, predicted = {}, {}, {}
                for rec, rs in samples.items():
                    times = rs[0]["times"]
                    matrix = np.asarray([r[channel] for r in rs])
                    half[rec] = batch_half_times(times, matrix.mean(axis=0), 0.1)[0]
                    resampled[rec] = batch_half_times(times, weights @ matrix, 0.1)
                    predicted[rec] = batch_half_times(times, rs[0][channel + "_pair"], 0.1)[0]
                values[channel] = half["opinion_random"] / half["random"]
                boots[channel] = resampled["opinion_random"] / resampled["random"]
                pairs[channel] = predicted["opinion_random"] / predicted["random"]
            values["differential"] = values["node"] / values["edge"]
            boots["differential"] = boots["node"] / boots["edge"]
            pairs["differential"] = pairs["node"] / pairs["edge"]
            for channel in values:
                valid = boots[channel][np.isfinite(boots[channel])]
                lo, hi = np.quantile(valid, (tail, 1 - tail))
                answer.append(
                    {
                        "dynamics": d,
                        "initial": initial,
                        "quantity": channel,
                        "micro_factor": float(values[channel]),
                        "lower": float(lo),
                        "upper": float(hi),
                        "pair_factor": float(pairs[channel]),
                        "valid_bootstrap_fraction": len(valid) / len(weights),
                        "scope": "secondary contrast; eight process-seed blocks conditional on one initial graph",
                    }
                )
    table(out / "recommendation_contrasts.csv", answer)
    return answer


def population(micro, root, protocol, out):
    """Compute manuscript population contrasts from matched independent initials and conditional process branches; write aggregate/control evidence."""
    selected = [
        r
        for r in micro
        if r["meta"]["dynamics"] == "hk"
        and r["meta"]["recommender"] == "random"
        and r["meta"]["alpha"] == 0.02
        and r["meta"]["ratio"] == 10
        and r["meta"]["b"] == 161
        and r["meta"]["quadrature"] == 33
        and r["meta"]["process_replicate"] < 4
    ]
    grouped = collections.defaultdict(list)
    for r in selected:
        grouped[(r["meta"]["n"], r["meta"]["initial_replicate"])].append(r)
    answer = []
    for (n, initial), rs in sorted(grouped.items()):
        assert len(rs) == 4
        times = rs[0]["times"]
        avg = np.asarray([r["rho"] for r in rs]).mean(axis=0)
        p = np.asarray(rs[0]["pair_rho"])
        boot = bootstrap_weights(4, protocol) @ np.asarray([r["rho"] for r in rs])
        w1 = np.abs(np.cumsum(avg / avg.sum() - p / p.sum())).sum() * 2 / len(avg)
        bw1 = np.abs(np.cumsum(boot / boot.sum(axis=1)[:, None] - p / p.sum(), axis=1)).sum(axis=1) * 2 / len(avg)
        low, high = np.quantile(bw1, (0.025, 0.975))
        row = {
            "n": n,
            "initial_seed_index": initial,
            "processes": 4,
            "mean_rho_w1": float(w1),
            "lower": float(low),
            "upper": float(high),
        }
        for channel in ("node", "edge"):
            curve = np.asarray([r[channel] for r in rs]).mean(axis=0)
            pair = rs[0][channel + "_pair"]
            a = [{"time": t, "e": e} for t, e in zip(times, curve)]
            b = [{"time": t, "e": e} for t, e in zip(times, pair)]
            row[channel + "_gap_of_mean_curve"] = trajectory_gap(a, b, "e")["gap"]
            row[channel + "_micro_tau50"] = 0.02 * batch_half_times(times, curve, 0.1)[0]
            row[channel + "_pair_tau50"] = 0.02 * batch_half_times(times, pair, 0.1)[0]
        answer.append(row)
    table(out / "population_equal_repeats.csv", answer)
    old = read_csv(root / "analysis" / "micro" / "population_four_processes.csv")
    aggregates = []
    for n in (500, 1000, 2000):
        rs = [r for r in old if int(r["n"]) == n]
        assert len(rs) == 16
        row = {"n": n, "initials": 4, "processes_per_initial": 4}
        for metric in (
            "node_w1",
            "node_energy_gap",
            "edge_energy_gap",
            "node_T50_relative_error",
            "edge_T50_relative_error",
        ):
            row[metric + "_median"], row[metric + "_max"] = map(
                float, np.percentile([number(r, metric) for r in rs], (50, 100))
            )
        aggregates.append(row)
    table(out / "population_aggregate.csv", aggregates)
    return answer, aggregates, selected
