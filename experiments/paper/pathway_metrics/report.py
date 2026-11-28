"""Offline SQLite summaries and figures for a frozen I_w/P-E comparison study.

The comparison source is read-only. Fixed study/selection JSON files are design
inputs; analysis tables and provenance are committed to a separate SQLite store
before any CSV or figure export. No raw trajectory or I_w calculation occurs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common.io.json import canonical, fingerprint, sha
from common.storage.documents import DocumentStore, DocumentWrite
from ehk.metrics import pathway
from .study import load_manifest

NAMESPACE = "ehk-iw-pe-comparison-v1"
REPORTS = "ehk-iw-pe-reports-v1"
TABLES = "ehk-iw-pe-report-tables-v1"
LABELS = ("PbS", "SbP", "simultaneous", "separation_only", "segregation_only", "neither", "unavailable")
DISPLAY_LABELS = ("P first", "E first", "Same observation", "P only", "E only", "Neither", "Unavailable")
CUTOFFS = (.4, .5, .55, .6, .65, .7, .8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(value):
    return value["origin"], value["name"]


def write_csv(path, rows):
    require(bool(rows), f"No summary rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strata(rows):
    yield "overall", "all", rows
    algorithms = ("StructureM9", "OpinionM9")
    reposts = (0., .1, .25, .5)
    signs = ("D<0", "D=0", "D>0")
    for dimension, values, key in (("algorithm", algorithms, "algorithm"), ("p", reposts, "p"), ("D_sign", signs, "D_sign")):
        for value in values:
            yield dimension, str(value), [r for r in rows if r[key] == value]
    for algorithm, p, sign in itertools.product(algorithms, reposts, signs):
        yield "algorithm_p_D", f"{algorithm};p={p:g};{sign}", [r for r in rows if r["algorithm"] == algorithm and r["p"] == p and r["D_sign"] == sign]


def coverage(rows, design, tasks):
    """Expose unfinished, sequential coverage rather than only observed levels."""
    total = len(tasks)
    output = []
    dimensions = (("alpha", design["alpha_axis"]), ("q", design["q_axis"]),
                  ("p", design["p_axis"]), ("algorithm", design["algorithms"]),
                  ("origin", sorted({task[0] for task in tasks})))
    for dimension, levels in dimensions:
        for value in levels:
            planned = sum(task[0] == value for task in tasks) if dimension == "origin" else total // len(levels)
            complete = sum(row[dimension] == value for row in rows)
            require(complete <= planned, "Completed marginal coverage exceeds selected design")
            output.append(dict(dimension=dimension, level=value, selected_n=planned,
                               completed_n=complete, missing_or_failed_n=planned-complete,
                               completion_fraction=complete/planned))
    return output


def summaries(rows):
    counts, quantiles, thresholds, horizons = [], [], [], []
    for dimension, value, block in strata(rows):
        common = dict(dimension=dimension, stratum=value, stratum_n=len(block))
        for label in LABELS:
            selected = [r for r in block if r["label"] == label]
            n = len(selected)
            for group in ("low", "high"):
                group_n = sum((r["Iw"] >= .6) == (group == "high") for r in selected)
                counts.append(dict(**common, PE_label=label, PE_label_n=n, Iw_cutoff=.6, Iw_group=group, n=group_n, within_PE_row_fraction=group_n / n if n else None))
            values = np.array([r["Iw"] for r in selected], dtype=float)
            qs = np.quantile(values, (0, .05, .25, .5, .75, .95, 1)) if n else [None] * 7
            quantiles.append(dict(**common, PE_label=label, n=n, **dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"), qs))))
        for cutoff in CUTOFFS:
            table = Counter(("high" if r["Iw"] >= cutoff else "low", r["label"]) for r in block)
            both = sum(r["p_step"] is not None and r["e_step"] is not None for r in block)
            ordered = sum(r["label"] in ("PbS", "SbP") for r in block)
            record = dict(**common, cutoff=cutoff, low_n=sum(r["Iw"] < cutoff for r in block), high_n=sum(r["Iw"] >= cutoff for r in block), both_thresholds_observed_n=both, ordered_both_thresholds_n=ordered, simultaneous_n=sum(r["label"] == "simultaneous" for r in block))
            for label in LABELS:
                for group in ("low", "high"):
                    record[f"{group}_{label}_n"] = table[group, label]
            record["direction_corresponding_ordered_n"] = table["low", "PbS"] + table["high", "SbP"]
            record["direction_different_ordered_n"] = table["high", "PbS"] + table["low", "SbP"]
            require(record["direction_corresponding_ordered_n"] + record["direction_different_ordered_n"] == ordered, "Ordered counts do not sum")
            thresholds.append(record)
        for label in ("all", *LABELS):
            selected = block if label == "all" else [r for r in block if r["label"] == label]
            terminal = np.array([r["terminal"] for r in selected])
            horizons.append(dict(**common, PE_label=label, n=len(selected), min_terminal_step=int(terminal.min()) if len(selected) else None, median_terminal_step=float(np.median(terminal)) if len(selected) else None, max_terminal_step=int(terminal.max()) if len(selected) else None, at_simulation_cap_n=sum(r["terminal"] == r["cap"] for r in selected), median_observations=float(np.median([r["observations"] for r in selected])) if selected else None))
    endpoints = []
    for peak_class in ("one_KDE_peak", "multiple_KDE_peaks", "unavailable_peak_count"):
        for label in LABELS:
            selected = [r for r in rows if r["peak_class"] == peak_class and r["label"] == label]
            medians = {f"median_{key}": float(np.median([r[key] for r in selected])) if selected else None for key in ("terminal_P", "terminal_E", "terminal_I_p", "terminal_I_h", "Iw", "endpoint_completion")}
            endpoints.append(dict(terminal_peak_class=peak_class, PE_label=label, n=len(selected), Iw_cutoff=.6, low_n=sum(r["Iw"] < .6 for r in selected), high_n=sum(r["Iw"] >= .6 for r in selected), **medians))
    return dict(counts=counts, quantiles=quantiles, thresholds=thresholds, horizons=horizons, endpoints=endpoints)


def load_study(directory):
    directory = Path(directory).resolve()
    manifest = directory / "study.json"
    study = asdict(load_manifest(directory))
    selection_path = directory / study["selection_file"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(fingerprint(study["protocol"]) == study["protocol_id"], "Frozen protocol identity differs from its body")
    require(study["analysis_id"] == selection["design"]["analysis_id"], "Frozen primary analysis differs from selection")
    return study, selection, manifest


def read_snapshot(study):
    """Read exact frozen result keys in one transaction, without raw access."""
    selected, snapshot = [], []
    with DocumentStore(study["comparison_database"], readonly=True) as store:
        store.connection.execute("BEGIN")
        for task in study["tasks"]:
            key = task["result_key"]
            result_protocol = task.get("result_protocol_id") or study["protocol_id"]
            result = store.get(NAMESPACE, key)
            updated = store.connection.execute("SELECT updated_ns FROM documents WHERE namespace=? AND key=?", (NAMESPACE, key)).fetchone()
            snapshot.append((key, fingerprint(result), updated[0] if updated else None))
            if result is None:
                continue
            primary = result["primary"]
            require(result["protocol_id"] == result_protocol, "Result uses a different frozen task protocol")
            require(primary["identity"] == task["identity"] and fingerprint(primary) == task["primary_sha256"], "Result primary reference differs from frozen study")
            require(primary["source_fingerprint"] == task["source_fingerprint"], "Result raw identity differs from frozen study")
            require(fingerprint((result_protocol, primary)) == key, "Result does not match its frozen document key")
            require(result["state"] in ("complete", "failed"), "Unknown comparison state")
            selected.append(result)
    timestamps = [updated for _, _, updated in snapshot if updated is not None]
    return selected, dict(source_snapshot_sha256=fingerprint(snapshot), snapshot_max_updated_ns=max(timestamps) if timestamps else None)


def observations_for_report(results, study, selection):
    """Validate stored observations and derive scalar summaries, never I_w."""
    design, protocol = selection["design"], study["protocol"]
    require(protocol["observation_interval"] == design["pe_observation_interval"] == pathway.OBSERVATION_INTERVAL and protocol["include_terminal"] is True, "Unexpected saved observation cadence")
    require(protocol["separation_threshold"] == design["reference_pe_thresholds"]["P"] == pathway.SEPARATION_THRESHOLD and protocol["excess_threshold"] == design["reference_pe_thresholds"]["E"] == pathway.EXCESS_THRESHOLD, "Unexpected saved P/E thresholds")
    core = Path(pathway.__file__).resolve()
    recorded_core = [expected for path, expected in protocol["source_sha256"].items() if Path(path).resolve() == core]
    require(len(recorded_core) == 1 and sha(core) == recorded_core[0], "Current first_passage core differs from the collected protocol")
    rows = []
    for result in results:
        if result["state"] != "complete":
            continue
        primary = result["primary"]
        require(primary["analysis_id"] == study["analysis_id"] and primary["metric_revision"] == 6 and primary["endpoint_complement"] is True, "Unexpected primary I_w protocol")
        require(result["schema"] == 1 and result["error"] is None, "Malformed completed comparison")
        metadata = primary["metadata"]
        hp = metadata["HKParams"]
        alpha, q, p = hp["Influence"], hp["RewiringRate"], hp["RepostRate"]
        algorithm = metadata["RecsysFactoryType"]
        require(alpha in design["alpha_axis"] and q in design["q_axis"] and p in design["p_axis"] and algorithm in design["algorithms"] and metadata["PostRetainCount"] == design["post_retain_count"] == 0, "Comparison metadata lies outside selected design")
        terminal, cap = primary["terminal_step"], metadata["MaxSimulationStep"]
        require(0 <= terminal <= cap, "Terminal step outside saved horizon")
        observations = result["observations"]
        require([v["step"] for v in observations] == [*range(0, terminal, protocol["observation_interval"]), terminal], "Stored observation grid is incomplete")
        require(all(math.isfinite(v[k]) for v in observations for k in ("P", "E", "C_edge", "C_pair")), "Nonfinite saved P/E observation")
        passage = pathway.first_passage((v["step"], v["P"], v["E"]) for v in observations)
        require((passage.label, passage.separation_step, passage.segregation_step) == (result["observed_order"], result["separation_step"], result["segregation_step"]), "Stored order disagrees with saved P/E observations")
        require(passage.label in LABELS and all(math.isfinite(primary[key]) for key in ("I_w", "terminal_I_p", "terminal_I_h")), "Invalid recorded process summary")
        peaks = primary["opinion_peaks"]
        peak_class = "one_KDE_peak" if peaks == 1 else "multiple_KDE_peaks" if peaks is not None and peaks >= 2 else "unavailable_peak_count"
        rows.append(dict(origin=primary["identity"]["origin"], algorithm=algorithm, alpha=alpha, q=q, p=p, D_sign="D<0" if q < alpha else "D>0" if q > alpha else "D=0", Iw=primary["I_w"], label=passage.label, p_step=passage.separation_step, e_step=passage.segregation_step, terminal=terminal, cap=cap, observations=len(observations), peak_class=peak_class, terminal_P=observations[-1]["P"], terminal_E=observations[-1]["E"], terminal_I_p=primary["terminal_I_p"], terminal_I_h=primary["terminal_I_h"], endpoint_completion=(1-primary["terminal_I_p"])*primary["terminal_I_h"]))
    return rows


def plot_tables(tables, path, receipt):
    """Draw manuscript-style PDF/PNG from committed summary tables only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotting import PAPER_WIDTH, plt_figure, plt_save_and_close, setup_paper_params

    setup_paper_params()
    fig, axes = plt_figure(n_col=2, total_width=PAPER_WIDTH*(2/3), hw_ratio=.95,
                           gridspec_kw={"width_ratios": (1.3, 1)}, sharey=True)
    overall = {r["PE_label"]: r for r in tables["quantiles"] if r["dimension"] == "overall"}
    boxes, positions = [], []
    for i, label in enumerate(LABELS):
        values = overall[label]
        if values["n"]:
            boxes.append(dict(med=values["median"], q1=values["q25"], q3=values["q75"], whislo=values["min"], whishi=values["max"]))
            positions.append(i)
    axes[0].bxp(boxes, positions=positions, widths=.45, vert=False, showfliers=False,
                 manage_ticks=False, patch_artist=True,
                 boxprops={"facecolor": "#dce9f4", "edgecolor": "tab:blue", "linewidth": 1.2},
                 whiskerprops={"color": "tab:blue", "linewidth": 1.2},
                 capprops={"color": "tab:blue", "linewidth": 1.2},
                 medianprops={"color": "tab:orange", "linewidth": 1.4})
    axes[0].axvline(.6, color="0.3", linestyle="--", linewidth=1)
    axes[0].set_yticks(range(len(LABELS)), DISPLAY_LABELS)
    axes[0].set_xlabel(r"$I_w$")
    axes[0].set_title("(a)", loc="left")
    axes[0].set_xlim(min(0., min(box["whislo"] for box in boxes)-.03),
                     max(1.03, max(box["whishi"] for box in boxes)+.03))
    axes[0].grid(axis="x", alpha=.2)
    counts = {(r["PE_label"], r["Iw_group"]): r["n"] for r in tables["counts"] if r["dimension"] == "overall"}
    matrix = np.array([[counts[label, group] for group in ("low", "high")] for label in LABELS])
    shown = axes[1].pcolormesh(np.arange(3)-.5, np.arange(len(LABELS)+1)-.5,
                              matrix, cmap="Blues", vmin=0, shading="flat")
    axes[1].set_xticks((0, 1), (r"$I_w<0.6$", r"$I_w\geq0.6$"))
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    for i, values in enumerate(matrix):
        for j, count in enumerate(values):
            text = f"{count}\n({count / values.sum():.1%})" if values.sum() else "0"
            axes[1].text(j, i, text, ha="center", va="center", color="white" if count > matrix.max()*.55 else "black", fontsize=10)
    axes[1].set_title("(b)", loc="left")
    colorbar = fig.colorbar(shown, ax=axes[1], label="Count", shrink=.85, pad=.04)
    colorbar.solids.set_rasterized(False)
    axes[0].set_ylim(len(LABELS)-.5, -.5)
    plt_save_and_close(fig, str(path))
    sample = "Preliminary, incomplete fixed selection" if receipt["preliminary"] else "Complete fixed selection"
    caption = (
        f"{sample}: {receipt['complete']} of {receipt['planned']} tasks; "
        f"{receipt['missing']} missing and {receipt['failed']} failed. "
        "The historical pilot is separate. P/E categories describe first observed crossings of "
        f"P={receipt['protocol']['separation_threshold']:g} and E={receipt['protocol']['excess_threshold']:g}, "
        f"recorded every {receipt['protocol']['observation_interval']} completed steps and at the terminal step; "
        "same-observation crossings have unresolved within-interval order. "
        "Panel (a) shows minimum, lower quartile, median, upper quartile and maximum of the recorded I_w. "
        "The dashed line marks the working split at 0.6. Panel (b) shows counts and percentages within each P/E row; "
        "zero-size rows retain their category. These are separate process summaries, with neither treated as ground truth. "
        "The incomplete parameter coverage is reported in coverage.csv and must accompany interpretation.\n"
    )
    path.with_name(path.name+"_caption.txt").write_text(caption, encoding="utf-8")


def export_report(database, report_id, output, *, make_figure=True):
    """SQLite is the canonical analysis result; exports are generated exhibits."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with DocumentStore(database, readonly=True) as store:
        receipt = store.get(REPORTS, report_id)
        require(receipt is not None, "Report identity is absent from the analysis store")
        tables = {name: store.get(TABLES, key)["rows"] for name, key in receipt["tables"].items()}
    for name, table in tables.items():
        write_csv(output / f"{name}.csv", table)
    from common.plotting import core as plotting_core
    renderer = dict(
        analysis_receipt_sha256=fingerprint(receipt),
        renderer_source_sha256=sha(__file__),
        plotting_core_source_sha256=sha(plotting_core.__file__),
        render_parameters=dict(width_inches=plotting_core.PAPER_WIDTH*(2/3),
                               height_inches=plotting_core.PAPER_WIDTH*(2/3)/2*.95,
                               dpi=300, formats=["pdf", "png"],
                               typography="common.plotting.setup_paper_params",
                               rendered_figures=make_figure),
    )
    require(not (receipt.keys() & renderer.keys()), "Renderer fields would overwrite the analysis receipt")
    write_csv(output / "provenance.csv", [dict(property=k, value=canonical(v) if isinstance(v, (dict, list)) else v) for k, v in {**receipt, **renderer}.items()])
    if make_figure:
        plot_tables(tables, output / "comparison", receipt)


def analyze(study_directory, output, *, preliminary=False, make_figure=True):
    study, selection, manifest = load_study(study_directory)
    output = Path(output).resolve()
    database = output / "analysis.sqlite"
    require(database != Path(study["comparison_database"]).resolve(), "Report output must be separate from read-only comparison observations")
    results, snapshot = read_snapshot(study)
    counts = Counter(result["state"] for result in results)
    planned = len(study["tasks"])
    missing = planned-len(results)
    if not preliminary:
        require(counts["complete"] == planned and not counts["failed"] and not missing, f"Incomplete selected sample: complete={counts['complete']}, failed={counts['failed']}, missing={missing}; use --preliminary explicitly")
    require(counts["complete"] > 0, "No completed selected comparisons")
    rows = observations_for_report(results, study, selection)
    tasks = {identity(task) for task in selection["tasks"]}
    design = selection["design"]
    if not preliminary:
        expected = set(itertools.product(design["algorithms"], design["alpha_axis"], design["q_axis"], design["p_axis"]))
        require(Counter((r["algorithm"], r["alpha"], r["q"], r["p"]) for r in rows) == Counter(expected), "Expected one run per selected design cell and algorithm")
    tables = summaries(rows)
    tables["coverage"] = coverage(rows, design, tasks)
    protocol = study["protocol"]
    receipt = dict(
        schema=1, status="preliminary incomplete selected sample" if preliminary else "complete fixed selected sample", preliminary=preliminary,
        study=str(manifest.parent), study_sha256=sha(manifest), selection_sha256=study["selection_sha256"],
        comparison_database=str(Path(study["comparison_database"]).resolve()), protocol_id=study["protocol_id"], primary_analysis_id=study["analysis_id"],
        planned=planned, complete=len(rows), failed=counts["failed"], missing=missing, **snapshot,
        selected_origin_counts=dict(Counter(t[0] for t in tasks)), completed_PE_labels=dict(Counter(r["label"] for r in rows)),
        result_protocol_counts=dict(Counter(result["protocol_id"] for result in results)),
        study_migration=study.get("migration"),
        protocol=protocol, report_script_sha256=sha(__file__),
        report_versions=dict(python=platform.python_version(), numpy=np.__version__),
        current_source_sha256_matches={path: sha(path) == expected if Path(path).is_file() else None for path, expected in protocol["source_sha256"].items()},
        current_first_passage_core_verified=True,
        scope="Fixed parameter selection, one repeat per cell; historical pilot separate; no per-cell variance estimate or shared-seed claim",
        interpretation="P/E order and continuous I_w tendency describe different process features; direction counts use only separately ordered crossings of both thresholds",
        endpoint_interpretation="Terminal KDE peaks and P/E order are separate descriptions, neither is tendency ground truth; completion is an algebraic decomposition of recorded indices at the same T",
        source_access="read-only SQLite transaction; no raw histories and no I_w recalculation",
    )
    report_id = fingerprint(receipt)
    table_keys = {name: fingerprint((report_id, name)) for name in tables}
    receipt.update(report_id=report_id, tables=table_keys)
    output.mkdir(parents=True, exist_ok=True)
    writes = [DocumentWrite(TABLES, table_keys[name], dict(report_id=report_id, table=name, rows=table)) for name, table in tables.items()]
    with DocumentStore(database) as store:
        store.put_many([*writes, DocumentWrite(REPORTS, report_id, receipt)])
    export_report(database, report_id, output, make_figure=make_figure)
    return dict(report_id=report_id, analysis_database=str(database), output=str(output), complete=len(rows), planned=planned, failed=counts["failed"], missing=missing, preliminary=preliminary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    analysis = commands.add_parser("analyze", help="Validate saved comparisons and commit SQLite summaries")
    analysis.add_argument("--study", type=Path, required=True, help="Frozen study directory containing study.json and its fixed selection")
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--preliminary", action="store_true", help="Explicitly report an incomplete fixed selected sample")
    analysis.add_argument("--no-figure", action="store_true")
    export = commands.add_parser("export", help="Draw figures and CSV exhibits using only a derived analysis SQLite store")
    export.add_argument("--database", type=Path, required=True)
    export.add_argument("--report-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "analyze":
        print(canonical(analyze(args.study, args.output, preliminary=args.preliminary, make_figure=not args.no_figure)))
    else:
        export_report(args.database, args.report_id, args.output, make_figure=not args.no_figure)
        print(canonical(dict(report_id=args.report_id, output=str(args.output.resolve()), source="derived SQLite only")))


if __name__ == "__main__":
    main()
