"""Render the design-pooled bars alongside explicit Opinion-history controls."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from experiments.paper.analysis.selection import descriptive_iw_cutoff, read_statistics
from experiments.paper.paths import PATHWAY_OUTPUT, statistics_database
from experiments.paper.pathway_index.bar_plot import figure5_axis_limits, plot_grouped_bars, summarize_bar, write_bar_caption
from experiments.paper.pathway_index.grouped_summary import HISTORY_SCOPES, _metrics, history_in_scope

setup_paper_params()


k_filters_1 = [
    (lambda row: row.recsys_type == "StructureM9", lambda row: row.repost_rate == 0),
    (lambda row: row.recsys_type == "StructureM9", lambda row: row.repost_rate != 0),
    (lambda row: row.recsys_type == "OpinionM9", lambda row: row.repost_rate == 0),
    (lambda row: row.recsys_type == "OpinionM9", lambda row: row.repost_rate != 0),
]
k_filters_1 = [(*filters, lambda row: history_in_scope(row.recsys_type, row.post_retain_count, "original-design"))
               for filters in k_filters_1]

k_labels_1 = ["St, !R", "St, R", "Op, !R", "Op, R"]

k_filters_2 = [
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 0,
        lambda row: row.repost_rate == 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 2,
        lambda row: row.repost_rate == 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 6,
        lambda row: row.repost_rate == 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 0,
        lambda row: row.repost_rate != 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 2,
        lambda row: row.repost_rate != 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 6,
        lambda row: row.repost_rate != 0,
    ),
]

k_labels_2 = ["k=0, !R", "k=2", "k=6", "k=0, R", "k=2", "k=6"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=statistics_database())
    parser.add_argument("--origin", action="append", help="Repeat to explicitly pool named sources.")
    parser.add_argument("--analysis-id")
    parser.add_argument("--output", type=Path, default=PATHWAY_OUTPUT.resolve())
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--history-scope", choices=HISTORY_SCOPES, default="original-design")
    args = parser.parse_args(argv)
    session = read_statistics(args.database, mode="gradation", origins=args.origin, analysis_id=args.analysis_id)
    cutoff = descriptive_iw_cutoff() if args.cutoff is None else args.cutoff
    scoped = [row for row in session if history_in_scope(row.recsys_type, row.post_retain_count, args.history_scope)]
    # The fifth panel is a history control, including when the other panels use zero.
    plot_matched = partial(plot_grouped_bars, records=scoped, cutoff=cutoff,
                           category_filters=k_filters_1, category_labels=k_labels_1)
    plot_history = partial(plot_grouped_bars, records=session, cutoff=cutoff,
                           category_filters=k_filters_2, category_labels=k_labels_2)

    fig, axes = plt_figure(n_row=2, n_col=3, constrained_layout=False)
    ax1, ax2 = axes[0][0], axes[0][1]
    ax3, ax4, ax5 = axes[1][0], axes[1][1], axes[1][2]

    axes[0][2].set_visible(False)

    # Use exactly the same four metric definitions as the formal Figure 5.
    evaluated = {(row.origin, row.name): _metrics(row) for row in session}

    def metric(name):
        return lambda row: evaluated[(row.origin, row.name)][name]

    f_ev_count = metric("rewiring_events")
    f_ev_step = metric("normalized_rewiring_time")
    f_auc_subj_pol_index = metric("subjective_index_auc")
    f_triads = metric("closed_triads")

    data_1 = plot_matched(ax1, metric=f_ev_count, plot_title="Event Count")
    ax1.set_title("(a) #rewiring events", loc="left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    data_2 = plot_matched(ax2, metric=f_ev_step, plot_title="Event Step")
    ax2.set_title("(b) norm. occurrence time", loc="left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    data_3 = plot_matched(ax3, f_auc_subj_pol_index, plot_title="AUC of Subjective Polarization Index")
    ax3.set_title(r"(c) AUC of $I_s$", loc="left")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    data_4 = plot_matched(ax4, metric=f_triads, plot_title="Triads")
    ax4.set_title("(d) #triads", loc="left")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    data_5 = plot_history(ax5, metric=f_triads, plot_title="Triads (Opinion)")
    ax5.set_title("(e) #triads, Op.", loc="left")
    ax5.grid(True, linestyle="--", alpha=0.5)
    ax5.legend()

    def limits_for(rows, filters, name):
        summaries = []
        for category in filters:
            for high in (False, True):
                def group(row, high=high):
                    value = row.pathway_integral
                    return value is not None and np.isfinite(value) and (value >= cutoff if high else value < cutoff)
                block = summarize_bar(rows, (*category, group), metric(name))
                summaries.append(dict(metric=name, n_observed=block.count, mean=block.mean, population_std=block.std))
        return figure5_axis_limits(pd.DataFrame(summaries))[name]

    for axis, name, rows, filters in (
        (ax1, "rewiring_events", scoped, k_filters_1),
        (ax2, "normalized_rewiring_time", scoped, k_filters_1),
        (ax3, "subjective_index_auc", scoped, k_filters_1),
        (ax4, "closed_triads", scoped, k_filters_1),
        (ax5, "closed_triads", session, k_filters_2),
    ):
        limits = limits_for(rows, filters, name)
        axis.set_yscale(limits["scale"])
        axis.set_ylim(*limits["limits"])
    fig.tight_layout()
    output = args.output / "zero" if args.history_scope == "zero" else args.output
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt_save_and_close(fig, str(figure_dir / "f_grad_index_interpret_raw"))
    details = [f"Panels a-d history scope: {args.history_scope}; selected n={len(scoped)}. "
               "StructureM9 kh=0; OpinionM9 kh=0" + (",2,6." if args.history_scope == "original-design" else "."),
               "Panel e always shows separate OpinionM9 kh=0/2/6 history controls from the available sources.",
               "Each run has weight one; pooled conditions follow observed counts. Comparisons are design-set means, not paired effects.",
               "Axes include finite mean +/- population SD; logarithmic axes require positive endpoints."]
    write_bar_caption(figure_dir / "f_grad_index_interpret_raw", session, cutoff,
                      details + data_1 + data_2 + data_3 + data_4 + data_5)

    with (figure_dir / "f_grad_index_interpret_raw_data.txt").open("w") as f:
        f.write(f"Available runs: {len(session)}; panels a-d scope: {args.history_scope}; "
                f"descriptive I_w cutoff: {cutoff:g}; counts=low/high\n")
        for line in data_1 + data_2 + data_3 + data_4 + data_5:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
