"""Render formal Figure 5 for an explicit history design and I_w cutoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.plotting import plt_save_and_close
from experiments.paper.analysis.selection import descriptive_iw_cutoff, read_statistics
from experiments.paper.paths import PATHWAY_OUTPUT, statistics_database
from experiments.paper.pathway_index.bar_plot import (
    figure5_summary_details, render_figure5, write_figure5_caption,
)
from experiments.paper.pathway_index.grouped_summary import HISTORY_SCOPES, history_in_scope, summarize


# Retained category selectors use the formal original-design collection by default.
# Metric evaluation and run aggregation belong exclusively to summarize().
k_filters_recsys = [
    (lambda row: row.recsys_type == "StructureM9",
     lambda row: history_in_scope(row.recsys_type, row.post_retain_count, "original-design")),
    (lambda row: row.recsys_type == "OpinionM9",
     lambda row: history_in_scope(row.recsys_type, row.post_retain_count, "original-design")),
]
k_labels_recsys = ["St.", "Op."]
k_filters_retweet = [
    (lambda row: row.repost_rate == 0,
     lambda row: history_in_scope(row.recsys_type, row.post_retain_count, "original-design")),
    (lambda row: row.repost_rate != 0,
     lambda row: history_in_scope(row.recsys_type, row.post_retain_count, "original-design")),
]
k_labels_retweet = ["!R", "R"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=statistics_database())
    parser.add_argument("--output", type=Path,
                        help="Output root; figures/ for the main scope, zero/figures/ for zero scope")
    parser.add_argument("--analysis-id")
    parser.add_argument("--origin", action="append", dest="origins",
                        help="Explicit source to include; repeat to select multiple sources")
    parser.add_argument("--cutoff", type=float, help="Descriptive I_w split; otherwise require EHK_IW_CUTOFF")
    parser.add_argument("--history-scope", choices=HISTORY_SCOPES, default="original-design")
    args = parser.parse_args(argv)
    records = read_statistics(args.database, mode="gradation", analysis_id=args.analysis_id,
                              origins=tuple(args.origins) if args.origins else None)
    cutoff = descriptive_iw_cutoff() if args.cutoff is None else args.cutoff
    data, summary, contrasts = summarize(records, cutoff, history_scope=args.history_scope)
    figure, _ = render_figure5(summary)
    output = args.output if args.output is not None else PATHWAY_OUTPUT.resolve()
    if args.history_scope == "zero":
        output = output / "zero"
    output = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "f_grad_index_interpret"
    plt_save_and_close(figure, str(stem))
    summary.to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    contrasts.to_csv(stem.with_name(stem.name + "_contrasts.csv"), index=False)
    write_figure5_caption(stem, records, data, summary, cutoff, history_scope=args.history_scope)
    details = [f"Available runs: {len(records)}; selected runs: {len(data)}; "
               f"history scope: {args.history_scope}; descriptive I_w cutoff: {cutoff:g}; counts=low/high"]
    stem.with_name(stem.name + "_data.txt").write_text(
        "\n".join(details + figure5_summary_details(summary)) + "\n", encoding="utf-8")
    return {"runs": len(data), "history_scope": args.history_scope, "cutoff": cutoff, "output": str(stem)}


if __name__ == "__main__":
    main()
