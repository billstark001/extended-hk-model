"""Study-owned statistics and display data derived from portable observations."""

import csv
import itertools
from collections import defaultdict
from contextlib import ExitStack

import numpy as np

from common.statistics import bootstrap_weights
from theory.variation import with_loio

from .bundle import csv_stream


class Tables:
    def __init__(self, root):
        self.root, self.stack, self.writers = root, ExitStack(), {}

    def __enter__(self):
        return self

    def add(self, name, row):
        if name not in self.writers:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.stack.enter_context(csv_stream(path, "wt"))
            writer = csv.DictWriter(handle, list(row))
            writer.writeheader()
            self.writers[name] = writer
        self.writers[name].writerow(row)

    def __exit__(self, *args):
        self.stack.close()


def matched_bands(records, protocol, tables):
    groups = defaultdict(list)
    for record in records:
        t = record["task"]
        if "matched" in t["roles"]:
            groups[(tuple(t["parameters"].items()), t["initial"])].append(record)
    for members in groups.values():
        members.sort(key=lambda r: r["task"]["branch"])
        t, first = members[0]["task"], members[0]["result"]
        p = t["parameters"]
        tau = np.array([f["time"] for f in first["frames"]]) * p["alpha"]
        pt = np.array([f["time"] for f in first["pair"]["frames"]]) * p["alpha"]
        for record in members:
            if not np.array_equal(tau, np.array([f["time"] for f in record["result"]["frames"]]) * p["alpha"]):
                raise ValueError("matched cohort time grids differ")
        weights = bootstrap_weights(len(members), protocol)
        quantile = (1 - protocol["confidence_level"]) / 2
        for channel in ("node", "edge"):
            values = np.array([[f["exact_" + channel + "_energy"] for f in r["result"]["frames"]] for r in members])
            low, high = np.quantile(weights @ values, [quantile, 1 - quantile], axis=0)
            pair = np.interp(tau, pt, [f[channel + "_energy"] for f in first["pair"]["frames"]])
            for i, at in enumerate(tau):
                tables.add(
                    "matched_curves.csv",
                    {
                        **p,
                        "initial": t["initial"],
                        "processes": len(members),
                        "channel": channel,
                        "tau": float(at),
                        "mean": float(values[:, i].mean()),
                        "lower": float(low[i]),
                        "upper": float(high[i]),
                        "pair": float(pair[i]),
                        "confidence_level": protocol["confidence_level"],
                        "uncertainty": "conditional process bootstrap",
                    },
                )


def representative_fields(records, protocol, tables):
    groups = defaultdict(list)
    for record in records:
        t, p = record["task"], record["task"]["parameters"]
        if (
            ("population" in t["roles"] or "network_snapshot" in t["roles"])
            and p["dynamics"] == "hk"
            and p["recommender"] == "opinion_random"
            and p["steepness"] == 4
            and p["population"] == protocol["matched_population"]
            and t["initial"] == 0
        ):
            groups[tuple(p.items())].append(record)
    for members in groups.values():
        p = members[0]["task"]["parameters"]
        at = protocol["main_snapshot_tau"]

        def snapshot(frames):
            hits = [f for f in frames if np.asarray(f.get("edge", [])).size and np.isclose(f["time"] * p["alpha"], at)]
            if not hits:
                raise ValueError(f"representative snapshot tau={at} was not recorded")
            return hits[0]

        # The same conditional population ensemble supplies the four release curves.
        # Normalize the mean raw energy, rather than averaging normalized branches.
        first = members[0]["result"]
        tau = np.array([f["time"] for f in first["frames"]]) * p["alpha"]
        pair_tau = np.array([f["time"] for f in first["pair"]["frames"]]) * p["alpha"]
        for channel in ("node", "edge"):
            values = np.array([[f["exact_" + channel + "_energy"] for f in r["result"]["frames"]] for r in members])
            for source, times, energy in (
                ("micro", tau, values.mean(0)),
                ("pair", pair_tau, np.array([f[channel + "_energy"] for f in first["pair"]["frames"]])),
            ):
                reference = float(np.median(energy[times >= 0.9 * times[-1]]))
                amplitude = float(energy[0] - reference)
                if amplitude <= 0:
                    raise ValueError("representative release requires positive amplitude")
                for time, raw in zip(times, energy):
                    tables.add(
                        "representative_response.csv",
                        {
                            "epsilon": p["epsilon"],
                            "source": source,
                            "channel": channel,
                            "tau": float(time),
                            "energy": float(raw),
                            "reference": reference,
                            "remaining": float((raw - reference) / amplitude),
                            "population": p["population"],
                            "initial": 0,
                            "processes": len(members),
                            "horizon_tau": float(times[-1]),
                        },
                    )
        fs = [snapshot(r["result"]["frames"]) for r in members]
        pair = snapshot(members[0]["result"]["pair"]["frames"])
        bins = len(pair["rho"])
        dx = 2 / bins
        axis = -1 + (np.arange(bins) + 0.5) * dx
        for source, rho, edge in (
            (
                "micro",
                np.mean([f["rho"] for f in fs], axis=0),
                np.mean([np.asarray(f["edge"]).reshape(bins, bins) for f in fs], axis=0),
            ),
            ("pair", np.asarray(pair["rho"]), np.asarray(pair["edge"]).reshape(bins, bins)),
        ):
            rho, edge = rho / rho.sum(), edge / edge.sum()
            md = {
                "epsilon": p["epsilon"],
                "source": source,
                "tau": at,
                "population": p["population"],
                "initial": 0,
                "processes": len(members),
                "bins": bins,
            }
            for i, x in enumerate(axis):
                tables.add(
                    "representative_nodes.csv",
                    {**md, "bin": i, "x": float(x), "mass": float(rho[i]), "density": float(rho[i] / dx)},
                )
                for j, y in enumerate(axis):
                    tables.add(
                        "representative_edges.csv.gz",
                        {
                            **md,
                            "i": i,
                            "j": j,
                            "x": float(x),
                            "y": float(y),
                            "mass": float(edge[i, j]),
                            "density": float(edge[i, j] / dx**2),
                        },
                    )


def population_variation(records, tables):
    groups = defaultdict(list)
    for record in records:
        t = record["task"]
        if "population" in t["roles"]:
            groups[tuple(t["parameters"].items())].append(record)
    for members in groups.values():
        p = members[0]["task"]["parameters"]
        initials = sorted({r["task"]["initial"] for r in members})
        cohorts = [
            sorted([r for r in members if r["task"]["initial"] == g], key=lambda r: r["task"]["branch"])
            for g in initials
        ]
        if len(initials) < 2 or min(map(len, cohorts)) < 2 or len(set(map(len, cohorts))) != 1:
            continue
        times = None
        frames, predictions = [], []
        for cohort in cohorts:
            group = []
            for r in cohort:
                values = {
                    round(f["time"] * p["alpha"], 10): f
                    for f in r["result"]["frames"]
                    if np.asarray(f.get("edge", [])).size
                }
                if times is None:
                    times = sorted(values)
                if sorted(values) != times:
                    raise ValueError("population field snapshot grids differ")
                group.append(values)
            frames.append(group)
            predictions.append(
                {
                    round(f["time"] * p["alpha"], 10): f
                    for f in cohort[0]["result"]["pair"]["frames"]
                    if np.asarray(f.get("edge", [])).size
                }
            )
        draws = np.array(list(itertools.product(range(len(cohorts)), repeat=len(cohorts))))
        assert times is not None
        for at in times:
            yfields, pfields = [], []
            yscalars, pscalars = [], []
            for g, cohort in enumerate(cohorts):
                pf = predictions[g][at]
                prho = np.asarray(pf["rho"])
                prho = prho / prho.sum()
                bins = len(prho)
                dx = 2 / bins
                pe = np.asarray(pf["edge"]).reshape(bins, bins)
                pe = pe / pe.sum()
                x = -1 + (np.arange(bins) + 0.5) * dx
                mask = np.abs(x[:, None] - x[None, :]) <= p["epsilon"]

                def excess(e):
                    return float(np.sum((e - np.outer(e.sum(1), e.sum(0))) * mask))

                pfields.append((np.cumsum(prho), np.cumsum(np.cumsum(pe, axis=0), axis=1).ravel()))
                pscalars.append((pf["variance"], excess(pe)))
                yf, ys = [], []
                for b, r in enumerate(cohort):
                    f = frames[g][b][at]
                    rho = np.asarray(f["rho"])
                    rho = rho / rho.sum()
                    edge = np.asarray(f["edge"]).reshape(bins, bins)
                    edge = edge / edge.sum()
                    yf.append((np.cumsum(rho), np.cumsum(np.cumsum(edge, axis=0), axis=1).ravel()))
                    ys.append((f["exact_variance"], excess(edge)))
                    tables.add(
                        "population_outcomes.csv",
                        {
                            **p,
                            "tau": at,
                            "initial": initials[g],
                            "branch": r["task"]["branch"],
                            "variance": f["exact_variance"],
                            "near_consensus_1e8": f["exact_variance"] < 1e-8,
                            "near_consensus_1e6": f["exact_variance"] < 1e-6,
                            "near_consensus_1e4": f["exact_variance"] < 1e-4,
                        },
                    )
                yfields.append(yf)
                yscalars.append(ys)
            for i, metric in enumerate(("node_cdf", "edge_cdf")):
                y = np.array([[v[i] for v in group] for group in yfields])
                prediction = np.array([v[i] for v in pfields])
                stat = with_loio(y, prediction, dx if i == 0 else dx**2)
                if not np.isclose(
                    stat["individual_mse"],
                    stat["process_variance"] + stat["conditional_bias_squared_corrected"],
                    atol=1e-12,
                    rtol=1e-10,
                ):
                    raise ValueError("MSE != W+B²")
                tables.add(
                    "population_variation.csv",
                    {
                        **p,
                        "tau": at,
                        "metric": metric,
                        "initials": len(cohorts),
                        "processes_per_initial": len(cohorts[0]),
                        **stat,
                        "uncertainty": "LOIO range; not a confidence interval",
                    },
                )
            scalar_values = np.array(yscalars)
            ps = np.array(pscalars)
            for i, metric in enumerate(("opinion_variance", "excess_concordance")):
                means = scalar_values[:, :, i].mean(1)
                estimates = means[draws].mean(1)
                low, high = np.quantile(estimates, [0.025, 0.975])
                tables.add(
                    "population_structure.csv",
                    {
                        **p,
                        "tau": at,
                        "metric": metric,
                        "mean": float(means.mean()),
                        "lower": float(low),
                        "upper": float(high),
                        "initials": len(cohorts),
                        "processes_per_initial": len(cohorts[0]),
                        "uncertainty": "exploratory 95% whole-initial-group bootstrap",
                        **with_loio(scalar_values[:, :, i : i + 1], ps[:, i : i + 1]),
                    },
                )
