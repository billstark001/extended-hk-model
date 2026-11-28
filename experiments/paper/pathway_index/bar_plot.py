"""Finite, counted summaries for the descriptive pathway bar figures."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import StatisticFilter, source_counts
from experiments.paper.pathway_index.grouped_summary import HISTORY_SCOPES, METRICS

FIGURE5_AXIS_LABELS = ("Rewiring events", "Normalized event time", r"AUC of $I_s$", "Closed triads")


@dataclass(frozen=True)
class BarSummary:
    count: int
    mean: float
    std: float


def summarize_bar(
    records: Iterable[ScenarioStatistics],
    filters: Sequence[StatisticFilter],
    metric: Callable[[ScenarioStatistics], float],
) -> BarSummary:
    values = np.asarray([
        metric(row) for row in records
        if row.name.startswith("s_grad") and all(predicate(row) for predicate in filters)
    ], dtype=float)
    values = values[np.isfinite(values)]
    return BarSummary(len(values), float(values.mean()), float(values.std())) if values.size else BarSummary(0, math.nan, math.nan)


def plot_grouped_bars(
    ax: Axes,
    metric: Callable[[ScenarioStatistics], float],
    *,
    records: Sequence[ScenarioStatistics],
    cutoff: float,
    category_filters: Sequence[Sequence[StatisticFilter]],
    category_labels: Sequence[str],
    plot_title: str = "",
) -> list[str]:
    """Capture the cutoff once; never read configuration inside row predicates.

    Exported counts are finite observations of this metric, in low/high I_w order.
    Missing groups remain missing rather than being rendered as zero bars.
    """
    if not math.isfinite(cutoff):
        raise ValueError("cutoff must be finite")
    if not category_filters or len(category_filters) != len(category_labels):
        raise ValueError("category filters and labels must have equal nonzero lengths")
    groups = (
        lambda row: row.pathway_integral is not None and math.isfinite(row.pathway_integral) and row.pathway_integral < cutoff,
        lambda row: row.pathway_integral is not None and math.isfinite(row.pathway_integral) and row.pathway_integral >= cutoff,
    )
    labels = (r"low $I_w$", r"high $I_w$")
    summaries = [
        [summarize_bar(records, (*filters, group), metric) for group in groups]
        for filters in category_filters
    ]
    lines = [f"\n=== {plot_title} ==="] if plot_title else []
    for category, blocks in zip(category_labels, summaries):
        for label, summary in zip(labels, blocks):
            lines.append(f"{category} - {label}: {summary.mean:.4f} ± {summary.std:.4f} (n={summary.count})")

    width = 0.3
    x = np.arange(len(category_labels))
    for group, (label, color) in enumerate(zip(labels, ("tab:blue", "tab:red"))):
        available = np.asarray([blocks[group].count > 0 for blocks in summaries])
        ax.bar(
            (x + group * width)[available],
            [blocks[group].mean for blocks in summaries if blocks[group].count],
            width,
            yerr=[blocks[group].std for blocks in summaries if blocks[group].count],
            label=label, capsize=4, edgecolor="black", color=color, alpha=0.5,
        )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(category_labels)
    return lines


def write_bar_caption(stem: Path, records: Sequence[ScenarioStatistics], cutoff: float, details: Sequence[str]) -> None:
    """Keep sample counts in the exported caption, outside the figure artwork."""
    header = (
        f"Available completed trajectories: n={len(records)}. "
        f"Descriptive low/high I_w split at {cutoff:g}; not first-passage labels. "
        "Bars show means with population standard deviations. "
        "Each n counts finite observations of the named metric; empty groups have no bar.\n"
        "Data-dependent display limits include every finite mean+/-SD endpoint.\n"
    )
    if records:
        header += ("Source counts: " + "; ".join(f"{origin}: n={count}" for origin, count in source_counts(records).items())
                   + "; analysis_id: " + ", ".join(sorted({row.analysis_id for row in records})) + ".\n")
    stem.with_name(stem.name + "_caption.txt").write_text(header + "\n".join(details) + "\n", encoding="utf-8")


def figure5_axis_limits(summary) -> dict:
    """Include every observed mean +/- population SD without fixed paper limits.

    Accepting concatenated summaries permits matched axes across cutoffs. Counts
    and signed endpoints are retained; a nonpositive endpoint requires a linear
    axis rather than clipping an error bar to use a logarithmic scale.
    """
    limits = {}
    for metric in METRICS:
        block = summary[(summary["metric"] == metric) & (summary["n_observed"] > 0)]
        endpoints = np.asarray([
            (row["mean"] - row["population_std"], row["mean"] + row["population_std"])
            for row in block.to_dict("records")
            if math.isfinite(row["mean"]) and math.isfinite(row["population_std"])
        ], dtype=float)
        if not endpoints.size:
            limits[metric] = dict(scale="linear", limits=[0., 1.])
            continue
        lower, upper = float(endpoints[:, 0].min()), float(endpoints[:, 1].max())
        if metric in ("rewiring_events", "closed_triads") and lower > 0:
            limits[metric] = dict(scale="log", limits=[lower / 1.15, upper * 1.15])
        else:
            lower, upper = min(0., lower), max(0., upper)
            span = upper - lower or 1.
            limits[metric] = dict(scale="linear", limits=[lower - .04 * span, upper + .10 * span])
    return limits


def render_figure5(summary, *, limits=None):
    """Render the sole Figure 5 layout from grouped_summary's evaluated metrics."""
    from common.plotting import PAPER_WIDTH, plt_figure, setup_paper_params

    setup_paper_params()
    limits = figure5_axis_limits(summary) if limits is None else limits
    fig, axes = plt_figure(n_row=2, n_col=4, total_width=PAPER_WIDTH * (2 / 3), hw_ratio=1.35,
                           sharey="col", constrained_layout=False)
    index = {(row["dimension"], row["category"], row["group"], row["metric"]): row
             for row in summary.to_dict("records")}
    for row, (dimension, categories, labels) in enumerate((
        ("policy", ("StructureM9", "OpinionM9"), ("St.", "Op.")),
        ("reposting", ("off", "on"), ("!R", "R")),
    )):
        for col, (metric, ylabel) in enumerate(zip(METRICS, FIGURE5_AXIS_LABELS)):
            ax = axes[row, col]
            for offset, (group, color) in enumerate((("low", "tab:blue"), ("high", "tab:red"))):
                available = [(position, index.get((dimension, category, group, metric)))
                             for position, category in enumerate(categories)]
                available = [(position, record) for position, record in available
                             if record is not None and record["n_observed"] > 0
                             and math.isfinite(record["mean"]) and math.isfinite(record["population_std"])]
                ax.bar([position + (offset - .5) * .30 for position, _ in available],
                       [record["mean"] for _, record in available], width=.30,
                       yerr=[record["population_std"] for _, record in available], color=color, alpha=.55,
                       edgecolor="black", linewidth=.7, capsize=2,
                       error_kw=dict(elinewidth=1, capthick=1), label=rf"{group} $I_w$")
            ax.set_xticks(np.arange(2), labels)
            ax.set_ylabel(ylabel)
            ax.set_title(f"({chr(97 + row * 4 + col)})", loc="left")
            ax.set_yscale(limits[metric]["scale"])
            ax.set_ylim(*limits[metric]["limits"])
            if limits[metric]["scale"] == "linear" and metric in ("rewiring_events", "closed_triads"):
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.grid(axis="y", alpha=.25, linestyle="--")
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=1.4, w_pad=.7)
    return fig, axes


def figure5_summary_details(summary) -> list[str]:
    lines = []
    for row in summary.to_dict("records"):
        value = (f"{row['mean']:.6g} +/- {row['population_std']:.6g}"
                 if row['n_observed'] else "missing")
        lines.append(f"{row['dimension']}:{row['category']} - {row['group']} - {row['metric']}: "
                     f"{value} (n={row['n_observed']}; selected runs={row['n_runs']})")
    return lines


def write_figure5_caption(stem: Path, records, data, summary, cutoff: float, *, history_scope: str) -> None:
    """Record the actual design, configuration weights and metric counts outside artwork."""
    if history_scope not in HISTORY_SCOPES:
        raise ValueError(f"unknown history scope: {history_scope}")
    selected = {(row["origin"], row["name"]) for row in data.to_dict("records")}
    rows = [row for row in records if (row.origin, row.name) in selected]
    configurations = Counter((row.recsys_type, row.post_retain_count, row.repost_rate) for row in rows)
    history = "StructureM9 kh=0; OpinionM9 kh=0" + (",2,6" if history_scope == "original-design" else "")
    header = (
        f"History scope: {history_scope}; {history}. Selected completed trajectories: n={len(data)} "
        f"from {len(records)} available trajectories.\n"
        f"Descriptive low I_w<{cutoff:g} and high I_w>={cutoff:g}; not first-passage labels. "
        "Top row: StructureM9 and OpinionM9; bottom row: p=0 (!R) and pooled p>0 (R).\n"
        "Each selected run has weight one; pooled history and p conditions follow their observed run counts, "
        "rather than equal configuration weights. Bars show means and population standard deviations (ddof=0), "
        "not confidence intervals. Each n counts finite observations of that metric; missing groups have no bar.\n"
        "Event time is event_step_mean/active_step for positive activity time; the event mean covers the stored "
        "whole trajectory. AUC integrates the stored I_s curve over normalized time [0,1], using the existing "
        "clipped piecewise-linear trapezoid function. Nonpositive/missing activity time leaves timing and AUC missing.\n"
        "Data-dependent axes include every finite mean+/-SD endpoint; logarithmic axes are used only when all "
        "these endpoints are positive. No fixed historical y-limit is imposed.\n"
        "Selected source counts: " + "; ".join(f"{origin}: n={count}" for origin, count in source_counts(rows).items()) + "; "
        f"analysis_id: {', '.join(sorted({row.analysis_id for row in rows}))}.\n"
        "Selected configuration counts (policy, kh, p):\n"
    )
    configuration_lines = [f"{policy}, kh={kh}, p={p:g}: n={n}"
                           for (policy, kh, p), n in sorted(configurations.items())]
    stem.with_name(stem.name + "_caption.txt").write_text(
        header + "\n".join(configuration_lines + figure5_summary_details(summary)) + "\n", encoding="utf-8")
