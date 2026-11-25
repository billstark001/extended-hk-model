"""Render recorded storage measurements without running benchmark actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from statistics import median

BACKENDS = ("current", "tinymongo", "indexed_split")
ACTIONS = (
    "load",
    "point_metadata",
    "point_full",
    "selective_metadata",
    "scan_metadata",
    "scan_full",
    "nested_metadata",
    "state_update",
    "replace_full",
)
QUERY_ACTIONS = {"selective_metadata", "scan_metadata", "scan_full", "nested_metadata", "scope_metadata", "scope_full"}
ACTION_NAMES = {
    "load": "Load put / document",
    "point_metadata": "Key read: metadata / operation",
    "point_full": "Key read: full document / operation",
    "selective_metadata": "Filtered metadata / query",
    "scan_metadata": "Metadata scan / query",
    "scan_full": "Full-document scan / query",
    "nested_metadata": "Nested metadata filter / query",
    "state_update": "State update / document",
    "replace_full": "Full-document replacement / operation",
}
PLOT_NAMES = {
    "load": "Load: put/document",
    "point_metadata": "Point: metadata",
    "point_full": "Point: full",
    "selective_metadata": "Selective: metadata",
    "scan_metadata": "Scan: metadata",
    "scan_full": "Scan: full",
    "nested_metadata": "Nested: metadata",
    "state_update": "State update",
    "replace_full": "Full replacement",
}
COLORS = {"current": "#4477AA", "tinymongo": "#EE7733", "indexed_split": "#228833", "sqlite_optimized": "#7755AA"}
MIB = 1024**2


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _values(values):
    return [number for value in values if (number := _number(value)) is not None]


def _median(values):
    values = _values(values)
    return median(values) if values else None


def _quantile(values, probability):
    values = sorted(_values(values))
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def _format(value, digits=3, scale=1):
    number = _number(value)
    return "—" if number is None else f"{number / scale:,.{digits}f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _kind(info):
    kind = info.get("data_kind", "not provided")
    if "synthetic" in kind:
        return "Calibrated synthetic data; values are not simulation results"
    if "transformed_actual" in kind:
        return "Transformed completed summaries with original values preserved"
    return str(kind)


def _read_records(directory):
    records, excluded, warnings = [], [], []
    for path in sorted((directory / "results").glob("*.json")):
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict):
                raise ValueError("worker result is not an object")
        except (OSError, ValueError) as error:
            warnings.append(f"{path.name}: {error}")
            continue
        record = {**record, "_file": path.name}
        if any(not record.get(key) for key in ("backend", "profile", "action")):
            warnings.append(f"{path.name}: missing backend/profile/action; excluded from the summary")
            continue
        if record.get("status") != "ok":
            excluded.append(record)
        else:
            records.append(record)
    # A retried worker must not become a second independent process repetition.
    identities = {}
    for record in records + excluded:
        repeat = record.get("repeat", record["_file"])
        key = (record["profile"], record["backend"], record["action"], str(repeat))
        identities.setdefault(key, []).append(record)
    duplicates = {key for key, items in identities.items() if len(items) > 1}
    if duplicates:
        warnings.extend(
            f"Duplicate worker identity {_cell(key)}; all corresponding files were excluded from numeric summaries"
            for key in sorted(duplicates)
        )
        records = [
            r
            for r in records
            if (r["profile"], r["backend"], r["action"], str(r.get("repeat", r["_file"]))) not in duplicates
        ]
        excluded = [
            r
            for r in excluded
            if (r["profile"], r["backend"], r["action"], str(r.get("repeat", r["_file"]))) not in duplicates
        ]
    return records, excluded, warnings


def _phases(record):
    raw = record.get("operation_seconds")
    if isinstance(raw, list) and raw and len(_values(raw)) == len(raw):
        raw = _values(raw)
        if record["action"] in QUERY_ACTIONS:
            yield "first", raw[:1]
            if len(raw) > 1:
                yield "steady", raw[1:]
        else:
            yield "all", raw
    else:
        # Aggregate p50 values cannot reconstruct the first and subsequent calls.
        yield "unpartitioned", None


def summarize(records, plan):
    """One row per action/phase; process repetitions remain the aggregation unit."""
    grouped = {}
    for record in records:
        for phase, samples in _phases(record):
            key = (record["profile"], record["backend"], record["action"], phase)
            grouped.setdefault(key, []).append((record, samples))
    rows = []
    for (profile, backend, action, phase), entries in sorted(grouped.items()):
        process_p50, process_p95, process_seconds, operation_counts = [], [], [], []
        for record, samples in entries:
            latency = record.get("latency") or {}
            process_p50.append(
                _quantile(samples, 0.5) * 1000 if samples is not None else _number(latency.get("p50_ms"))
            )
            process_p95.append(
                _quantile(samples, 0.95) * 1000 if samples is not None else _number(latency.get("p95_ms"))
            )
            process_seconds.append(sum(samples) if samples is not None else _number(latency.get("seconds")))
            operation_counts.append(len(samples) if samples is not None else _number(latency.get("operations")))
        members = [r for r, _ in entries]
        info = plan.get("profiles", {}).get(profile, {})
        counts = _values(r.get("count") for r in members)
        p50 = _values(process_p50)
        deltas = [
            max(0, peak - base)
            for r in members
            if (peak := _number(r.get("peak_rss_action_bytes"))) is not None
            and (base := _number(r.get("baseline_rss_bytes"))) is not None
        ]
        row = {
            "profile": profile,
            "data_kind": info.get("data_kind"),
            "backend": backend,
            "action": action,
            "phase": phase,
            "successful_processes": len(entries),
            "planned_repetitions": plan.get("repetitions", 3),
            "profile_count": info.get("count"),
            "profile_measured_count": info.get("measured_count"),
            "worker_count_min": min(counts) if counts else None,
            "worker_count_max": max(counts) if counts else None,
            "operation_count": sum(_values(operation_counts)) if _values(operation_counts) else None,
            "process_seconds_median": _median(process_seconds),
            "process_seconds_min": min(_values(process_seconds)) if _values(process_seconds) else None,
            "process_seconds_max": max(_values(process_seconds)) if _values(process_seconds) else None,
            "operation_p50_ms_median": _median(p50),
            "operation_p50_ms_min": min(p50) if p50 else None,
            "operation_p50_ms_max": max(p50) if p50 else None,
            "operation_p95_ms_median": _median(process_p95),
            "end_to_end_seconds_median": _median(r.get("end_to_end_seconds") for r in members),
            "baseline_rss_bytes_median": _median(r.get("baseline_rss_bytes") for r in members),
            "action_peak_rss_bytes_median": _median(r.get("peak_rss_action_bytes") for r in members),
            "action_peak_rss_delta_bytes_median": _median(deltas),
            "process_peak_rss_bytes_median": _median(r.get("peak_rss_bytes") for r in members),
            "parent_monitored_peak_rss_bytes_median": _median(
                r.get("parent_monitored_peak_rss_bytes") for r in members
            ),
            "files_apparent_bytes_median": _median((r.get("files") or {}).get("apparent_bytes") for r in members),
            "files_allocated_bytes_median": _median((r.get("files") or {}).get("allocated_bytes") for r in members),
            "open_seconds_median": _median(r.get("open_seconds") for r in members),
            "checkpoint_seconds_median": _median(r.get("checkpoint_seconds") for r in members),
            "source_files": json.dumps([r["_file"] for r in members]),
        }
        rows.append(row)
    return rows


def _action_order(actions):
    return sorted(actions, key=lambda action: (ACTIONS.index(action) if action in ACTIONS else len(ACTIONS), action))


def _figures(directory, rows, records, profiles, backends):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    profiles = profiles or ["no_measurements"]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(
        1, len(profiles), squeeze=False, figsize=(max(5, 4.9 * len(profiles)), 7.4), constrained_layout=True
    )
    keys = sorted(
        {(r["action"], r["phase"]) for r in rows},
        key=lambda key: (
            _action_order({r["action"] for r in rows}).index(key[0]),
            {"all": 0, "first": 1, "steady": 2, "unpartitioned": 3}.get(key[1], 4),
        ),
    )
    lookup = {(r["profile"], r["backend"], r["action"], r["phase"]): r for r in rows}
    bar_width = 0.75 / max(1, len(backends))
    for column, profile in enumerate(profiles):
        axis = axes[0, column]
        for index, backend in enumerate(backends):
            for location, (action, phase) in enumerate(keys):
                row = lookup.get((profile, backend, action, phase), {})
                value = row.get("operation_p50_ms_median")
                if value is None or value <= 0:
                    continue
                low = row.get("operation_p50_ms_min", value)
                high = row.get("operation_p50_ms_max", value)
                position = location + (index - (len(backends) - 1) / 2) * bar_width
                axis.barh(position, value, height=bar_width * 0.88, color=COLORS.get(backend, "#888888"))
                axis.errorbar(
                    value,
                    position,
                    xerr=[[max(0, value - low)], [max(0, high - value)]],
                    color="#333333",
                    capsize=2,
                    linewidth=0.7,
                )
        labels = [
            PLOT_NAMES.get(a, a)
            + ({"first": " / first", "steady": " / subsequent", "unpartitioned": " / unsplit"}.get(p, ""))
            for a, p in keys
        ]
        axis.set_yticks(np.arange(len(keys)), labels if column == 0 else [""] * len(keys))
        axis.invert_yaxis()
        if any(patch.get_width() > 0 for patch in axis.patches):
            axis.set_xscale("log")
        else:
            axis.text(0.5, 0.5, "No positive latency measurements", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(profile)
        axis.set_xlabel("Median process p50 (ms; log scale)")
        axis.grid(axis="x", alpha=0.2)
    figure.legend(
        handles=[Patch(color=COLORS.get(b, "#888888"), label=b) for b in backends],
        loc="outside lower center",
        ncol=max(1, len(backends)),
        frameon=False,
    )
    figure.suptitle("Operation latency; whiskers show process p50 min/max")
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"performance.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)

    actions = _action_order({r["action"] for r in records})
    figure, axes = plt.subplots(
        2, len(profiles), squeeze=False, figsize=(max(5, 4.9 * len(profiles)), 9), constrained_layout=True
    )
    for column, profile in enumerate(profiles):
        for metric_row, delta in enumerate((False, True)):
            axis = axes[metric_row, column]
            for index, backend in enumerate(backends):
                for location, action in enumerate(actions):
                    selected = [
                        r for r in records if (r["profile"], r["backend"], r["action"]) == (profile, backend, action)
                    ]
                    values = []
                    for record in selected:
                        peak, base = (
                            _number(record.get("peak_rss_action_bytes")),
                            _number(record.get("baseline_rss_bytes")),
                        )
                        if peak is not None and (not delta or base is not None):
                            values.append(max(0, peak - base) / MIB if delta else peak / MIB)
                    if not values:
                        continue
                    value = median(values)
                    position = location + (index - (len(backends) - 1) / 2) * bar_width
                    axis.barh(position, value, height=bar_width * 0.88, color=COLORS.get(backend, "#888888"))
                    axis.errorbar(
                        value,
                        position,
                        xerr=[[value - min(values)], [max(values) - value]],
                        color="#333333",
                        capsize=2,
                        linewidth=0.7,
                    )
            axis.set_yticks(
                np.arange(len(actions)), [PLOT_NAMES.get(a, a) for a in actions] if column == 0 else [""] * len(actions)
            )
            axis.invert_yaxis()
            axis.set_title(profile)
            axis.set_xlabel("Action peak RSS minus baseline (MiB)" if delta else "Sampled action peak RSS (MiB)")
            axis.grid(axis="x", alpha=0.2)
            if not axis.patches:
                axis.text(0.5, 0.5, "No action RSS measurements", ha="center", va="center", transform=axis.transAxes)
    figure.legend(
        handles=[Patch(color=COLORS.get(b, "#888888"), label=b) for b in backends],
        loc="outside lower center",
        ncol=max(1, len(backends)),
        frameon=False,
    )
    figure.suptitle("Whole-action sampled RSS; bars=median, whiskers=min/max across processes")
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"memory.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _recovery_rows(plan, directory):
    rows = []
    for backend, source in (plan.get("recovery") or {}).items():
        try:
            if isinstance(source, dict):
                result = source
            else:
                path = Path(source)
                if not path.is_absolute():
                    path = next((p for p in (directory / path, Path.cwd() / path) if p.is_file()), directory / path)
                result = json.loads(path.read_text())
            rows.append(
                (
                    backend,
                    result.get("passed_rounds"),
                    result.get("planned_rounds"),
                    result.get("all_passed"),
                    str(source),
                )
            )
        except (OSError, TypeError, ValueError) as error:
            rows.append((backend, None, None, "not read", str(error)))
    return rows


def render(run_dir):
    directory = Path(run_dir).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    records, excluded, warnings = _read_records(directory)
    rows = summarize(records, plan)
    profiles = list(plan.get("profiles", {})) or sorted({r["profile"] for r in records})
    backends = list(plan.get("backends", BACKENDS))
    repetitions = int(plan.get("repetitions", 3))
    actions = list(dict.fromkeys(("load", *plan.get("actions", ACTIONS))))
    expected = set(product(profiles, backends, actions, range(repetitions)))
    done = {
        (r["profile"], r["backend"], r["action"], r.get("repeat"))
        for r in records + excluded
        if r.get("status") in ("ok", "not_applicable")
    }
    missing = expected - done
    smoke = repetitions < 3 or any(
        info.get("measured_count", info.get("count")) != info.get("count") for info in plan.get("profiles", {}).values()
    )
    csv_path = directory / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = list(rows[0]) if rows else ["profile", "backend", "action", "phase", "successful_processes"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _figures(directory, rows, records, profiles, backends)

    lines = [
        "# Isolated storage benchmark results",
        "",
        "This report summarizes saved worker JSON without running new measurements. Numeric summaries include only status=ok results; failed and inapplicable workers are listed separately.",
        "",
    ]
    if smoke:
        lines.extend(
            [
                "This is a limited or single-repetition workflow check, not a full-scale comparison with independent process repetitions.",
                "",
            ]
        )
    lines.extend(
        [
            f"Planned independent process repetitions per action: {repetitions}; queries per process: {plan.get('query_repetitions', 'not provided')}. "
            f"Read {len(records)} successful workers and {len(excluded)} non-successful workers; {len(missing)} planned results are still missing.",
            "",
            "## Data and configuration",
            "",
            "| profile | Original task count | Measured document count | Data type |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for profile in profiles:
        info = plan.get("profiles", {}).get(profile, {})
        lines.append(
            f"| {_cell(profile)} | {_format(info.get('count'), 0)} | {_format(info.get('measured_count'), 0)} | {_cell(_kind(info))} |"
        )
    lines.extend(
        [
            "",
            "full_paper and versioned_2x are calibrated synthetic workloads, not recovered full-paper statistics. See [plan.json](plan.json) for sources and assumptions.",
            "",
            "All selected backends use SQLite. The comparison covers document interfaces, indexes, codecs, and array layouts. WAL and synchronous=FULL are controlled benchmark settings, not claims about unconfigured library defaults.",
            "",
            "Each action uses a fresh process, but the OS page cache is uncontrolled. first denotes its first query and steady denotes subsequent queries, not cold-disk reads. The default two-query run has one subsequent sample. Records without raw operation_seconds remain unpartitioned.",
            "",
            "## Latency",
            "",
            "Tables report the median process-level operation p50 in ms; parentheses show successful independent process counts. summary.csv also records process p50 ranges and median process p95 values. Operation samples are not pooled as independent repetitions. With one query sample per phase and process, p95/p99 do not estimate tail latency; compare independent process medians and observed ranges.",
            "",
        ]
    )
    lookup = {(r["profile"], r["backend"], r["action"], r["phase"]): r for r in rows}
    for profile in profiles:
        lines.extend(
            [
                f"### {_cell(profile)}",
                "",
                "| Operation and phase | " + " | ".join(backends) + " |",
                "| --- | " + " | ".join("---:" for _ in backends) + " |",
            ]
        )
        keys = sorted(
            {(r["action"], r["phase"]) for r in rows if r["profile"] == profile},
            key=lambda k: (actions.index(k[0]) if k[0] in actions else len(actions), k[1]),
        )
        for action, phase in keys:
            cells = []
            for backend in backends:
                row = lookup.get((profile, backend, action, phase))
                cells.append(
                    "—" if row is None else f"{_format(row['operation_p50_ms_median'])} ({row['successful_processes']})"
                )
            lines.append(f"| {_cell(ACTION_NAMES.get(action, action))} · {phase} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "![Operation latency](performance.png)",
            "",
            "Whiskers show the observed process p50 minimum and maximum, not confidence intervals. Log axes show positive latency values only.",
            "",
            "## Loading and file size",
            "",
            "Total put time sums per-document adapter calls, including encoding, compression, and commit but excluding input generation and hashing. Load-loop elapsed time also includes streamed generation and input hashes; it ends before spot checks and checkpointing. File sizes use worker measurements rather than estimates from logical array sizes.",
            "",
            "| profile / backend | Total put seconds | Load-loop elapsed seconds | Checkpoint seconds | Apparent file MiB | Allocated file MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["action"] == "load":
            lines.append(
                f"| {row['profile']} / {row['backend']} | {_format(row['process_seconds_median'])} | {_format(row['end_to_end_seconds_median'])} | {_format(row['checkpoint_seconds_median'])} | {_format(row['files_apparent_bytes_median'], scale=MIB)} | {_format(row['files_allocated_bytes_median'], scale=MIB)} |"
            )
    lines.extend(
        [
            "",
            "## Memory",
            "",
            "Action peaks use psutil RSS sampled every 10 ms, with a separate baseline. Process ru_maxrss includes initialization and other work and does not replace the action peak. Memory is not split between first and steady. Sampling includes untimed validation; load also includes generation and checkpointing, and state_update includes full-read validation. State-update RSS therefore measures the whole action, not the update alone. Metadata and full scans provide the main memory comparison. Values are process-repetition medians in MiB.",
            "",
            "| profile / backend / action | Baseline | Action peak | Above baseline | Process ru_maxrss |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    seen = set()
    for row in rows:
        identity = (row["profile"], row["backend"], row["action"])
        if identity in seen:
            continue
        seen.add(identity)
        lines.append(
            "| "
            + " / ".join(identity)
            + " | "
            + " | ".join(
                _format(row[field], scale=MIB)
                for field in (
                    "baseline_rss_bytes_median",
                    "action_peak_rss_bytes_median",
                    "action_peak_rss_delta_bytes_median",
                    "process_peak_rss_bytes_median",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "![Action memory](memory.png)",
            "",
            "## Inapplicable, failed, and missing results",
            "",
            "| profile / backend / action | status | File | Reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in excluded:
        reason = record.get("reason", record.get("error", "not provided"))
        lines.append(
            f"| {_cell(record['profile'])} / {_cell(record['backend'])} / {_cell(record['action'])} | {_cell(record.get('status', 'not provided'))} | {_cell(record['_file'])} | {_cell(reason)} |"
        )
    if not excluded:
        lines.append("| — | — | — | No non-successful results |")
    if missing:
        lines.extend(
            ["", f"There are {len(missing)} planned workers without completed results; this summary is partial."]
        )
    if warnings:
        lines.extend(["", "Read errors:", "", *[f"- {_cell(item)}" for item in warnings]])
    recovery = _recovery_rows(plan, directory)
    lines.extend(
        [
            "",
            "## Independent recovery validation",
            "",
            "Recovery validation is separate from performance timing. It tests writer-process SIGKILL, not power loss.",
            "",
            "| backend | Passed rounds | Planned rounds | all_passed |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for backend, passed, planned, all_passed, source in recovery:
        lines.append(f"| {_cell(backend)} | {_format(passed, 0)} | {_format(planned, 0)} | {_cell(all_passed)} |")
    if not recovery:
        lines.append("| — | — | — | No recovery results attached to the plan |")
    lines.extend(
        [
            "",
            "The plan recovery field retains report paths and source identities. Rendering does not rerun recovery tests.",
            "",
            "[Complete summary CSV](summary.csv) · [Latency PDF](performance.pdf) · [Memory PDF](memory.pdf)",
            "",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "run_dir": str(directory),
        "summary_rows": len(rows),
        "successful_workers": len(records),
        "excluded_workers": len(excluded),
        "missing_workers": len(missing),
        "warnings": warnings,
        "smoke_or_limited": smoke,
        "report": str(directory / "report.md"),
        "summary": str(csv_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
