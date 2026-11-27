"""Finite, counted summaries for the descriptive pathway bar figures."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import StatisticFilter


@dataclass(frozen=True)
class BarSummary:
    count: int
    mean: float
    std: float


@dataclass(frozen=True)
class PaperBarLimits:
    event_count: tuple[float, float] = (1e3, 1e4)
    event_time: tuple[float, float] = (0.0, 0.4)
    subjective_auc: tuple[float, float] = (0.3, 0.8)
    triads: tuple[float, float] = (1e3, 2e5)


PAPER_BAR_LIMITS = PaperBarLimits()


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
        "Display limits match the existing paper; the summaries retain values outside the displayed ranges.\n"
    )
    if records:
        header += f"Origin: {records[0].origin}; analysis_id: {records[0].analysis_id}.\n"
    stem.with_name(stem.name + "_caption.txt").write_text(header + "\n".join(details) + "\n", encoding="utf-8")
