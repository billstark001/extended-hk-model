"""Compare the recovered 32-run P/E pilot with recorded revision-6 I_w.

This reads both SQLite databases without modification. It reuses saved P/E
passage times after checking the recovered CSV, raw-source identity and metadata.
It neither reruns the probe nor recalculates I_w. Passage order is a separate
process summary, not ground truth for an accuracy score. The selected pilot is
kept separate from any expanded P/E sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

NAMESPACE = "ehk-paper-statistics-v3"
PILOT_ANALYSIS = "b494acedaead6a8fd9aaa7a2ac4e5c237801d66b738d53264dd21c754beba6fb"
PRIMARY_ANALYSIS = "e499b1ab5ff7ddc6480c5578127996e1945eecfd78af2f9a605f0fb7b93a267e"
CUTOFFS = (.4, .5, .6, .7, .8)
LABELS = ("PbS", "SbP", "separation_only", "segregation_only", "simultaneous", "neither", "unavailable")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compatible_metadata(old, new, prefix="metadata"):
    """Every historical declaration must match; additional new flags are fine."""
    if isinstance(old, dict):
        require(isinstance(new, dict), f"{prefix}: new value is not an object")
        for key, value in old.items():
            require(key in new, f"{prefix}.{key}: historical declaration missing")
            compatible_metadata(value, new[key], f"{prefix}.{key}")
    else:
        require(old == new, f"{prefix}: {old!r} != {new!r}")


def csv_step(value):
    if value == "":
        return None
    number = float(value)
    require(math.isfinite(number) and number.is_integer(), f"Invalid saved step: {value}")
    return int(number)


def plot_pilot(pairs, path):
    """Show only the verified 24 zero-history runs, separately from expansion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    codes = ("PbS", "SbP", "simultaneous", "separation_only", "segregation_only", "neither", "unavailable")
    labels = ("P first", "E first", "Same observation", "P only", "E only", "Neither", "Unavailable")
    selected = [(old, new) for old, new in pairs if old["post_retain_count"] == 0]
    require(len(selected) == 24, "Pilot figure requires the matched 24 runs")
    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    rng = np.random.default_rng(20261127)
    for i, code in enumerate(codes):
        values = [new["pathway_integral"] for old, new in selected if old["pathway_label"] == code]
        if values:
            ax.scatter(i+rng.uniform(-.12, .12, len(values)), values, s=28, alpha=.8, edgecolor="none")
            ax.plot([i-.19, i+.19], [np.median(values)]*2, color="black", linewidth=2)
    ax.axhline(.6, color="black", linestyle="--", linewidth=1, label="Working split at 0.6")
    ax.set_xticks(range(len(codes)), labels, rotation=25, ha="right")
    ax.set_ylabel(r"Recorded current $L$-based $I_w$")
    ax.set_xlabel("Separately saved P/E observation order")
    ax.set_title("Recovered parameter-selected pilot: 24 zero-history runs\nCurrent Iw; historical P/E times; separate from expanded sample", fontsize=11)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("D:/Research/extended-hk-model"))
    parser.add_argument("--pilot-database", type=Path)
    parser.add_argument("--database", type=Path, default=Path("E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite"))
    parser.add_argument("--diagnostics", type=Path, required=True, help="Frozen recovered 32-run diagnostics CSV")
    parser.add_argument("--output", type=Path, required=True, help="Explicit directory for compact pilot audit exports")
    parser.add_argument("--figure", action="store_true", help="Plot the verified zero-history pilot independently")
    args = parser.parse_args()
    pilot_database = args.pilot_database or args.model_root / "outputs/metric_revision_20261127/pilot/stats-v3.sqlite"
    sys.path[:0] = [str(args.model_root / "src"), str(args.model_root)]
    from common.storage.documents import DocumentStore
    from ehk.metrics.pathway import FirstPassage

    with DocumentStore(pilot_database, readonly=True) as store:
        store.connection.execute("BEGIN")
        pilot = list(store.values(NAMESPACE))
    require(len(pilot) == 32, "Expected the complete recovered 32-run pilot")
    require(len({r["name"] for r in pilot}) == 32, "Duplicate pilot run names")
    require(all(r["analysis_id"] == PILOT_ANALYSIS and r["metric_revision"] == 5 for r in pilot), "Wrong pilot protocol")
    require(all(r["origin"] == "F-data2-metric-pilot-20261127" for r in pilot), "Wrong pilot origin")
    with args.diagnostics.open(encoding="utf-8", newline="") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    diagnostic = {r["name"]: r for r in diagnostic_rows}
    require(len(diagnostic_rows) == len(diagnostic) == 32, "Expected 32 unique recovered CSV rows")
    require(set(diagnostic) == {r["name"] for r in pilot}, "Recovered CSV and pilot run names differ")

    names = set(diagnostic)
    current = {}
    with DocumentStore(args.database, readonly=True) as store:
        store.connection.execute("BEGIN")
        for row in store.values(NAMESPACE):
            if row["analysis_id"] == PRIMARY_ANALYSIS and row["origin"] == "original-data2" and row["name"] in names:
                require(row["name"] not in current, "Duplicate current run name")
                current[row["name"]] = row
    require(set(current) == names, "Not all historical runs exist in current primary data2")

    pairs = []
    for old in sorted(pilot, key=lambda r: r["name"]):
        new, saved = current[old["name"]], diagnostic[old["name"]]
        for row in (old, new):
            require(row["complete"] and not row["stale"], f"Incomplete/stale: {row['name']}")
            require(row["endpoint_complement"] is True, "Both I_w conventions must retain completion")
            require(math.isfinite(row["pathway_integral"]), "Nonfinite recorded I_w")
        require(new["metric_revision"] == 6, "Expected adopted revision 6")
        require(old["source"] == new["source"], f"Raw source object differs: {old['name']}")
        compatible_metadata(old["metadata"], new["metadata"], old["name"] + ".metadata")
        p, e = old["separation_step"], old["segregation_step"]
        require(csv_step(saved["separation_step"]) == p and csv_step(saved["segregation_step"]) == e, f"Recovered P/E times differ: {old['name']}")
        label = FirstPassage(p, e).label
        require(label == old["pathway_label"] == saved["first_passage"], f"Passage label mismatch: {old['name']}")
        require(float(saved["I_w"]) == old["pathway_integral"], f"Recovered revision-5 I_w differs: {old['name']}")
        pairs.append((old, new))

    counts, thresholds, migrations = [], [], []
    for cohort, selected in (("pilot_all32", pairs), ("pilot_kh0", [(o, n) for o, n in pairs if o["post_retain_count"] == 0])):
        require(len(selected) == (32 if cohort == "pilot_all32" else 24), "Unexpected cohort size")
        for cutoff in CUTOFFS:
            for revision, column in ((5, 0), (6, 1)):
                table = Counter(("high" if pair[column]["pathway_integral"] >= cutoff else "low", pair[0]["pathway_label"]) for pair in selected)
                for group in ("low", "high"):
                    for label in LABELS:
                        counts.append(dict(cohort=cohort, metric_revision=revision, cutoff=cutoff, Iw_group=group, saved_PE_label=label, n=table[group, label]))
                thresholds.append(dict(
                    cohort=cohort, metric_revision=revision, cutoff=cutoff, total_runs=len(selected),
                    low_n=sum(table["low", label] for label in LABELS), high_n=sum(table["high", label] for label in LABELS),
                    saved_order_PbS=sum(table[g, "PbS"] for g in ("low", "high")),
                    saved_order_SbP=sum(table[g, "SbP"] for g in ("low", "high")),
                    low_PbS=table["low", "PbS"], high_PbS=table["high", "PbS"],
                    low_SbP=table["low", "SbP"], high_SbP=table["high", "SbP"],
                    other_PE_labels=sum(table[g, label] for g in ("low", "high") for label in LABELS if label not in ("PbS", "SbP")),
                ))
            transitions = Counter(("high" if o["pathway_integral"] >= cutoff else "low", "high" if n["pathway_integral"] >= cutoff else "low") for o, n in selected)
            migrations.append(dict(cohort=cohort, cutoff=cutoff, total_runs=len(selected), low_to_low=transitions["low", "low"], low_to_high=transitions["low", "high"], high_to_low=transitions["high", "low"], high_to_high=transitions["high", "high"]))

    example_names = ("s_grad_sim0_rw0_dc3_rt1_st", "s_grad_sim0_rw3_dc3_rt1_st")
    examples = []
    for old, new in pairs:
        if old["name"] in example_names:
            examples.append(dict(name=old["name"], example="revision_change_at_0.6" if old["name"] == example_names[0] else "different_process_summaries", saved_PE_label=old["pathway_label"], saved_P_step=old["separation_step"], saved_E_step=old["segregation_step"], revision5_Iw=old["pathway_integral"], revision6_Iw=new["pathway_integral"], revision5_group="high" if old["pathway_integral"] >= .6 else "low", revision6_group="high" if new["pathway_integral"] >= .6 else "low", cutoff=.6))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("counts", counts), ("thresholds", thresholds), ("migration", migrations), ("examples", examples)):
        write_csv(args.output / f"recovered_pilot_{filename}.csv", data)
    provenance = {
        "scope": "parameter-selected pilot only; separate from expanded P/E observations",
        "pilot_database": str(pilot_database.resolve()), "current_database": str(args.database.resolve()),
        "pilot_analysis_id": PILOT_ANALYSIS, "current_analysis_id": PRIMARY_ANALYSIS,
        "current_origin": "original-data2", "pilot_runs": 32, "kh0_runs": 24,
        "equal_raw_source_objects": 32, "compatible_historical_metadata": 32,
        "recovered_CSV_times_and_labels_checked": 32,
        "saved_time_labels_checked_with": "ehk.metrics.pathway.FirstPassage.label",
        "recalculated_Iw": False, "reran_probe": False,
        "diagnostics_sha256": sha(args.diagnostics), "script_sha256": sha(__file__),
        "pathway_core_sha256": sha(args.model_root / "src/ehk/metrics/pathway.py"),
        "interpretation": "saved P/E order and I_w tendency are separate process summaries; no accuracy score",
    }
    write_csv(args.output / "recovered_pilot_provenance.csv", [dict(property=k, value=v) for k, v in provenance.items()])
    if args.figure:
        plot_pilot(pairs, args.output / "recovered_pilot_comparison.png")
    print(json.dumps({"checked_runs": 32, "kh0_runs": 24, "output": str(args.output.resolve()), "same_raw_sources": 32, "probe_rerun": False}, indent=2))


if __name__ == "__main__":
    main()
