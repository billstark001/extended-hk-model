"""Read-only audit of the historical I_w cutoff; outputs are explicit exhibits.

Run with the model repository's environment (NumPy/SciPy/Matplotlib installed).
No raw simulation, database mutation, or replacement state-metric route is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

PRIMARY = "e499b1ab5ff7ddc6480c5578127996e1945eecfd78af2f9a605f0fb7b93a267e"
NAMESPACE = "ehk-paper-statistics-v3"
CUTOFFS = (.4, .425, .5, .55, .6, .65, .7)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Explicit directory for compact audit exports")
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_root / "src"))
    sys.path.insert(0, str(args.model_root))
    from common.storage.documents import DocumentStore
    from ehk.metrics.trajectory import pathway_index
    from experiments.paper.pathway_index.data_utils import piecewise_linear_integral_trapz

    args.output.mkdir(parents=True, exist_ok=True)
    with DocumentStore(args.database, readonly=True) as store:
        store.connection.execute("BEGIN")
        records = list(store.values(NAMESPACE))
        max_update = store.connection.execute(
            "SELECT max(updated_ns) FROM documents WHERE namespace=?", (NAMESPACE,)
        ).fetchone()[0]
    rows = [d for d in records if d["analysis_id"] == PRIMARY]
    assert len(rows) == 24566, "Source cohort changed: review the audit scope"
    assert all(d["complete"] and not d["stale"] for d in rows)
    assert all(d["metric_revision"] == 6 and d["endpoint_complement"] is True for d in rows)
    protocol = rows[0]["protocol"]
    assert all(d["protocol"] == protocol for d in rows)
    assert protocol["metrics"]["distance_kde"]["bins"] == 4081
    assert set(d["origin"] for d in rows) == {"original-data2", "original-data3"}
    assert len({(d["origin"], d["name"]) for d in rows}) == len(rows)
    source_checks = {}
    for relative, expected in protocol["analysis_sources"].items():
        actual = hashlib.sha256((args.model_root / relative).read_bytes()).hexdigest()
        source_checks[relative] = dict(expected=expected, actual=actual, matches=actual == expected)
        # This audit uses recorded state arrays and the frozen scalar integration
        # core. It neither imports domain declarations nor regenerates a state.
        # Record that historical binding separately; every executed analysis and
        # metric source must still match. Do not claim whole-checkout equality.
        if relative != "src/ehk/domain.py":
            assert actual == expected, relative

    coverage = []
    for (origin, algorithm, kh), count in sorted(Counter(
        (d["origin"], d["recsys_type"], d["post_retain_count"]) for d in rows
    ).items()):
        coverage.append(dict(origin=origin, algorithm=algorithm, kh=kh, complete=count,
                             planned=3072, missing=3072-count))
    write_csv(args.output / "coverage.csv", coverage)

    iw = np.array([d["pathway_integral"] for d in rows])
    kh = np.array([d["post_retain_count"] for d in rows])
    algorithm = np.array([d["recsys_type"] for d in rows])
    origin = np.array([d["origin"] for d in rows])
    repost = np.array([d["repost_rate"] for d in rows])
    peaks = np.array([d["last_opinion_peak_count"] for d in rows])
    matched = kh == 0
    all_mask = np.ones(len(rows), dtype=bool)
    masks = {
        "all_history": all_mask,
        "kh0": matched,
        "kh0_data2": matched & (origin == "original-data2"),
        "kh0_data3": matched & (origin == "original-data3"),
        "kh0_OpinionM9": matched & (algorithm == "OpinionM9"),
        "kh0_StructureM9": matched & (algorithm == "StructureM9"),
        "kh0_p0": matched & (repost == 0),
        "kh0_p_positive": matched & (repost > 0),
        "kh0_one_peak": matched & (peaks == 1),
        "kh0_multiple_peaks": matched & (peaks >= 2),
    }
    quantiles = []
    for label, mask in masks.items():
        a = iw[mask]
        quantiles.append(dict(cohort=label, n=len(a), minimum=a.min(), q01=np.quantile(a,.01),
                              q25=np.quantile(a,.25), median=np.median(a), q75=np.quantile(a,.75),
                              q99=np.quantile(a,.99), maximum=a.max(), above_one=int((a>1).sum())))
    write_csv(args.output / "quantiles.csv", quantiles)

    # This KDE describes the distribution of scalar I_w across runs. It is
    # separate from the fixed 4081-bin state-distance KDE used to calculate I_w.
    grid = np.linspace(.1, 1.1, 1001)
    interior = (grid >= .3) & (grid <= .8)
    densities, curves = [], {}
    for label, mask in masks.items():
        for factor in (.5, 1., 2.):
            kde = gaussian_kde(iw[mask], bw_method=lambda obj: obj.scotts_factor()*factor)
            y = kde(grid)
            curves[label, factor] = y
            minimum = np.flatnonzero(interior)[np.argmin(y[interior])]
            modes = find_peaks(y, prominence=.025*y.max())[0]
            valleys = find_peaks(-y)[0]
            valleys = [j for j in valleys if interior[j]]
            densities.append(dict(
                cohort=label, n=int(mask.sum()), scott_multiplier=factor,
                bandwidth=float(np.sqrt(kde.covariance[0,0])),
                modes=";".join(f"{grid[j]:.3f}" for j in modes),
                interior_local_minima=";".join(f"{grid[j]:.3f}" for j in valleys),
                minimum_in_03_08=grid[minimum], density_minimum=y[minimum],
                density_at_06=float(kde([.6])[0]),
                density06_over_minimum=float(kde([.6])[0]/y[minimum]),
                interior_minimum_at_boundary=bool(minimum in np.flatnonzero(interior)[[0,-1]])))
    write_csv(args.output / "density.csv", densities)

    cuts = []
    for label, mask in masks.items():
        a = iw[mask]
        for cutoff in CUTOFFS:
            high = a >= cutoff
            cuts.append(dict(cohort=label, n=len(a), cutoff=cutoff, low_n=int((~high).sum()),
                             high_n=int(high.sum()), high_fraction=high.mean(),
                             changed_from_06=int((high != (a>=.6)).sum()),
                             changed_fraction_from_06=np.mean(high != (a>=.6))))
    write_csv(args.output / "cutoffs.csv", cuts)

    # Reuse the adopted trajectory core to verify the serialized values and to
    # describe (not remove) the agreed endpoint term.
    integral = np.array([pathway_index(d["p_index"], d["h_index"], endpoint_complement=False) for d in rows])
    completion = np.array([(1-d["p_index"][-1])*d["h_index"][-1] for d in rows])
    residual = np.abs(iw-integral-completion)
    assert residual.max() < 1e-12
    endpoint_rows = []
    for label in ("all_history", "kh0", "kh0_one_peak", "kh0_multiple_peaks"):
        mask = masks[label]
        endpoint_rows.append(dict(cohort=label, n=int(mask.sum()),
                                  integral_median=np.median(integral[mask]),
                                  completion_median=np.median(completion[mask]),
                                  iw_median=np.median(iw[mask]),
                                  max_serialization_residual=residual[mask].max()))
    write_csv(args.output / "endpoint.csv", endpoint_rows)

    count = np.array([d["event_count"] for d in rows], dtype=float)
    times = np.array([np.nan if d["event_step_mean"] is None else
                      d["event_step_mean"]/max(d["active_step"],1) for d in rows])
    triads = np.array([d["triads"] for d in rows], dtype=float)
    auc = np.array([piecewise_linear_integral_trapz(
        d["x_indices"]/max(d["active_step"],1), d["g_index"], 0, 1) for d in rows])
    effects = []
    for cohort in ("all_history", "kh0"):
        for cutoff in CUTOFFS:
            high = masks[cohort] & (iw >= cutoff)
            for comparison, base, target in (
                ("repost_positive_vs_zero", repost==0, repost>0),
                ("opinion_vs_structure", algorithm=="StructureM9", algorithm=="OpinionM9"),
            ):
                for metric, values in (("event_count",count), ("normalized_event_time",times),
                                       ("triads",triads), ("subjective_auc",auc)):
                    b = values[high & base & np.isfinite(values)]
                    t = values[high & target & np.isfinite(values)]
                    effects.append(dict(cohort=cohort, cutoff=cutoff, comparison=comparison,
                                        metric=metric, base_n=len(b), target_n=len(t),
                                        base_mean=b.mean(), target_mean=t.mean(),
                                        change_percent=100*(t.mean()/b.mean()-1)))
    write_csv(args.output / "effects.csv", effects)

    fig, axes = plt.subplots(2, 2, figsize=(10.5,7), constrained_layout=True)
    for axis, label, title in zip(axes[0], ("all_history","kh0"),
                                 ("All retained-history conditions", "Matched comparison: kh = 0")):
        axis.hist(iw[masks[label]], bins=np.arange(.1,1.101,.025), density=True,
                  color="0.85", label="Histogram")
        for factor, color in ((.5,"#5f9a76"),(1.,"#225ea8"),(2.,"#ae6b1f")):
            axis.plot(grid, curves[label,factor], color=color, label=f"Scott bandwidth x {factor:g}")
        axis.set_title(f"{title} (n = {masks[label].sum():,})",fontsize=11)
    for label, color, title in (("kh0_OpinionM9","#225ea8","OpinionM9"),
                                ("kh0_StructureM9","#d95f0e","StructureM9")):
        axes[1,0].plot(grid,curves[label,1.],color=color,label=title)
    axes[1,0].set_title("Matched comparison by algorithm",fontsize=11)
    for label, color, title in (("kh0_p0","#225ea8","p = 0"),
                                ("kh0_p_positive","#d95f0e","p > 0")):
        axes[1,1].plot(grid,curves[label,1.],color=color,label=title)
    axes[1,1].set_title("Matched comparison by repost condition",fontsize=11)
    for axis in axes.flat:
        axis.axvline(.6,color="black",ls="--",lw=1,label="Historical cutoff = 0.6")
        axis.set(xlabel="$I_w$",ylabel="Density",xlim=(.1,1.1))
        axis.legend(fontsize=8)
        axis.spines[["top","right"]].set_visible(False)
    fig.savefig(args.output / "density_audit.png",dpi=160)
    plt.close(fig)

    with args.database.open("rb") as stream:
        database_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    facts = [
        dict(item="database", value=str(args.database.resolve())),
        dict(item="namespace", value=NAMESPACE),
        dict(item="analysis_id", value=PRIMARY),
        dict(item="database_sha256", value=database_sha),
        dict(item="maximum_updated_ns", value=max_update),
        dict(item="documents_in_namespace", value=len(records)),
        dict(item="selected_documents", value=len(rows)),
        dict(item="excluded_other_analysis", value=len(records)-len(rows)),
        dict(item="matched_kh0", value=int(matched.sum())),
        dict(item="missing_first_passage_labels", value=sum(d["pathway_label"] is None for d in rows)),
        dict(item="analysis_source_checks", value=json.dumps(source_checks, sort_keys=True)),
        dict(item="nonexecuted_source_binding", value="src/ehk/domain.py is recorded but is not imported or used to regenerate states by this saved-array audit"),
        dict(item="audit_script_sha256", value=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        dict(item="python",value=sys.version.split()[0]),
        dict(item="numpy",value=np.__version__),
    ]
    write_csv(args.output / "provenance.csv", facts)
    print(f"Selected {len(rows)} records; kh0={matched.sum()}; excluded={len(records)-len(rows)}")
    print(f"Saved audit exhibits to {args.output.resolve()}")


if __name__ == "__main__":
    main()
