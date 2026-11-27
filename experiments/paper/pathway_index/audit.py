"""Audit raw metric conventions and the historical 0.6 low/high integral split."""

import argparse
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from common.io.json import atomic_json
from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.metrics.homophily import uniform_concordance_probability
from experiments.paper.analysis.records import StatisticsStore
from experiments.paper.analysis.selection import recorded_endpoint_complement
from experiments.paper.paths import PATHWAY_OUTPUT, statistics_database

GROUPS = ("Low I_w", "High I_w", "Unavailable")
CONVENTIONS = (
    "legacy_reconstructed",
    "correct_uniform_baseline",
    "use_edge_endpoints",
    "use_random_reference_distance",
    "use_js_divergence",
    "remove_endpoint_complement",
    "revised_full_trajectory",
)


def iw_group(value, cutoff=0.6):
    return GROUPS[2] if value is None or not math.isfinite(value) else GROUPS[int(value >= cutoff)]


def archived_scores(paths):
    """Optional read-only comparison with preserved historical wide tables."""
    scores = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT name, origin, grad_index FROM scenariostatistics WHERE grad_index IS NOT NULL")
            for name, origin, score in rows:
                key = name, origin
                if key in scores and scores[key] != score:
                    raise ValueError(f"conflicting archived scores for {name} ({origin})")
                scores[key] = score
    return scores


def analyze(database, legacy, output, origin=None, example=None, *, analysis_id=None, cutoff=0.6):
    if not math.isfinite(cutoff):
        raise ValueError("cutoff must be finite")
    scores = archived_scores(legacy)
    with StatisticsStore(database, readonly=True) as store:
        rows = store.records(origin=origin, analysis_id=analysis_id)
    if not rows:
        raise ValueError("no revised statistics available")
    if any(getattr(row, "stale", False) or getattr(row, "complete", True) is False for row in rows):
        raise ValueError(
            "selected statistics include stale or incomplete documents; complete reanalysis before auditing"
        )
    revisions = {row.analysis_id for row in rows}
    if len(revisions) != 1:
        raise ValueError("select one --analysis-id when multiple analysis revisions are present")
    endpoint_modes = {recorded_endpoint_complement(row) for row in rows}
    if len(endpoint_modes) != 1:
        raise ValueError("mixed endpoint-complement conventions in the selected analysis revision")
    endpoint_complement = next(iter(endpoint_modes))
    if len({row.origin for row in rows}) != 1:
        raise ValueError("select one --origin when multiple raw datasets are present")
    if len({row.name for row in rows}) != len(rows):
        raise ValueError("select one --origin when scenario names occur in multiple raw datasets")
    if not all(row.pathway_variants is not None for row in rows):
        raise ValueError("raw convention audit is missing; reanalyze without --skip-metric-audit")

    comparison, changes = [], []
    for row in rows:
        old = scores.get((row.name, row.origin))
        reconstructed = row.legacy_pathway_integral
        comparison.append(
            {
                "name": row.name,
                "origin": row.origin,
                "analysis_id": row.analysis_id,
                "endpoint_complement": endpoint_complement,
                "policy": row.recsys_type,
                "history": row.post_retain_count,
                "influence": row.decay,
                "rewiring": row.rewiring,
                "repost": row.repost_rate,
                "steps": row.step,
                "revised_iw": row.pathway_integral,
                "revised_group": iw_group(row.pathway_integral, cutoff),
                "full_trajectory_with_complement": row.pathway_variants.get("full_trajectory_with_complement"),
                "full_trajectory_without_complement": row.pathway_variants.get("full_trajectory_without_complement"),
                "reconstructed_legacy_iw": reconstructed,
                "reconstructed_group": iw_group(reconstructed, cutoff),
                "archived_iw": old,
                "archived_group": iw_group(old, cutoff),
                "reconstruction_prefix_start": getattr(row, "audit_prefix_start", None),
                "reconstruction_missing_observations": len(getattr(row, "audit_missing_steps", [])),
                "opinion_peaks": row.last_opinion_peak_count,
                "optional_pe_first_passage": row.pathway_label,
            }
        )
        previous = None
        for convention in CONVENTIONS:
            value = row.pathway_variants[convention]
            changes.append(
                {
                    "name": row.name,
                    "origin": row.origin,
                    "endpoint_complement": endpoint_complement,
                    "convention": convention,
                    "I_w": value,
                    "change_from_previous_convention": None if previous is None else value - previous,
                    "group_at_historical_cutoff": iw_group(value, cutoff),
                    "prefix_start": 0
                    if convention == "revised_full_trajectory"
                    else getattr(row, "audit_prefix_start", None),
                }
            )
            previous = value

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(comparison)
    data.to_csv(output / "pathway_audit.csv", index=False)
    pd.DataFrame(changes).to_csv(output / "metric_convention_changes.csv", index=False)
    counts = (
        pd.crosstab(data["reconstructed_group"], data["revised_group"])
        .reindex(index=GROUPS, columns=GROUPS, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    counts.to_csv(output / "pathway_group_migration.csv")
    report = {
        "schema": 4,
        "analysis_id": next(iter(revisions)),
        "endpoint_complement": endpoint_complement,
        "endpoint_convention": "unrecorded" if endpoint_complement is None else (
            "integral + (1-I_p(T))*I_h(T)" if endpoint_complement else "observed integral only"
        ),
        "full_trajectory_comparison": "completed and uncompleted values exported when recorded; never inferred",
        "records": len(rows),
        "cutoff": cutoff,
        "cutoff_status": "historical descriptive split; not a validated PbS/SbP classifier",
        "legacy_reconstruction": "same raw trajectory and deterministic KDE samples; revised adaptive grid",
        "missing_post_observations": "never imputed; sequential convention comparisons use the common finite prefix",
        "archived_scores_available": int(data.archived_iw.notna().sum()),
        "partial_legacy_reconstructions": int((data.reconstruction_missing_observations > 0).sum()),
        "full_campaign_validated": False,
    }
    atomic_json(output / "audit.json", report)

    setup_paper_params()
    fig, ax = plt_figure(total_width=6)
    grid = counts.to_numpy()
    ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set(
        xticks=np.arange(len(GROUPS)),
        xticklabels=GROUPS,
        yticks=np.arange(len(GROUPS)),
        yticklabels=GROUPS,
        xlabel="Revised pathway-index group",
        ylabel="Reconstructed historical group",
    )
    for (row_number, col), count in np.ndenumerate(grid):
        ax.text(
            col, row_number, str(count), ha="center", va="center", color="white" if count > grid.max() / 2 else "black"
        )
    plt_save_and_close(fig, str(output / "pathway_audit"))

    eligible = [row for row in rows if (example is None or row.name == example) and row.rho_edge is not None]
    if example is not None and not eligible:
        raise ValueError(f"no revised concordance trajectory for {example}")
    if eligible:
        row = eligible[0]
        fig, ax = plt_figure(total_width=7, hw_ratio=0.55)
        ax.plot(row.x_indices, row.rho_edge, label="Follow-edge endpoints")
        if row.rho_post is not None:
            ax.plot(row.x_indices, row.rho_post, label="Followee posts shown during the update")
        ax.axhline(uniform_concordance_probability(row.tolerance), ls="--", color="0.5", label="Uniform reference")
        ax.set(xlabel="Completed step", ylabel="Agent-weighted concordance", title=row.name, ylim=(0, 1.02))
        ax.legend()
        plt_save_and_close(fig, str(output / "edge_and_post_concordance"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=statistics_database())
    parser.add_argument(
        "--legacy", type=Path, action="append", help="Optional archived wide-table database, read only."
    )
    parser.add_argument("--origin", help="Only compare records from this workspace origin.")
    parser.add_argument("--analysis-id", help="Select a single metric protocol when several are stored.")
    parser.add_argument(
        "--cutoff", type=float, default=0.6, help="Descriptive historical split to audit, without validation claims."
    )
    parser.add_argument("--example", help="Scenario name for the edge and post concordance plot.")
    parser.add_argument("--output", type=Path, default=PATHWAY_OUTPUT.resolve() / "revision")
    args = parser.parse_args(argv)
    return analyze(
        args.database,
        args.legacy or [],
        args.output,
        args.origin,
        args.example,
        analysis_id=args.analysis_id,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    main()
