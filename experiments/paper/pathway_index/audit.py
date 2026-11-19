"""Compare revised first-passage labels with archived pathway scores."""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from common.storage.sqlite import create_db_engine_and_session
from ehk.metrics.homophily import uniform_concordance_probability
from ehk.metrics.pathway import PATHWAY_LABELS
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.paths import PATHWAY_OUTPUT, STAT_INPUT, statistics_database

LEGACY_GROUPS = ("PbS-like", "SbP-like", "No archived score")


def archived_scores(paths):
    scores = {}
    for path in paths:
        if not path.is_file():
            continue
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT name, origin, grad_index FROM scenariostatistics WHERE grad_index IS NOT NULL")
            for name, origin, score in rows:
                key = name, origin
                if key in scores and scores[key] != score:
                    raise ValueError(f"conflicting archived scores for {name} ({origin})")
                scores[key] = score
    return scores


def analyze(database, legacy, output, origin=None, example=None):
    scores = archived_scores(legacy)
    if not database.is_file():
        raise FileNotFoundError(database)
    engine, session = create_db_engine_and_session(str(database))
    try:
        query = select(ScenarioStatistics).where(ScenarioStatistics.pathway_label.is_not(None))
        if origin is not None:
            query = query.where(ScenarioStatistics.origin == origin)
        rows = session.scalars(query.order_by(ScenarioStatistics.name, ScenarioStatistics.origin)).all()
    finally:
        session.close()
        engine.dispose()
    if not rows:
        raise ValueError("no revised statistics available")

    comparison = []
    for row in rows:
        old = scores.get((row.name, row.origin))
        old_group = LEGACY_GROUPS[2] if old is None else LEGACY_GROUPS[int(old >= 0.6)]
        comparison.append(
            {
                "name": row.name,
                "origin": row.origin,
                "policy": row.recsys_type,
                "history": row.post_retain_count,
                "influence": row.decay,
                "rewiring": row.rewiring,
                "repost": row.repost_rate,
                "steps": row.step,
                "separation_step": row.separation_step,
                "segregation_step": row.segregation_step,
                "first_passage": row.pathway_label,
                "revised_iw": row.pathway_integral,
                "archived_iw": old,
                "archived_group": old_group,
                "opinion_peaks": row.last_opinion_peak_count,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(comparison)
    data.to_csv(output / "pathway_audit.csv", index=False)
    counts = pd.crosstab(data["first_passage"], data["archived_group"]).reindex(
        index=PATHWAY_LABELS, columns=LEGACY_GROUPS, fill_value=0
    )
    counts.to_csv(output / "pathway_counts.csv")

    setup_paper_params()
    fig, ax = plt_figure(total_width=6)
    grid = counts.to_numpy()
    ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set(
        xticks=np.arange(len(LEGACY_GROUPS)),
        xticklabels=LEGACY_GROUPS,
        yticks=np.arange(len(PATHWAY_LABELS)),
        yticklabels=[name.replace("_", " ") for name in PATHWAY_LABELS],
        xlabel="Archived pathway group",
        ylabel="First observed passage",
    )
    for (row, col), count in np.ndenumerate(grid):
        ax.text(col, row, str(count), ha="center", va="center", color="white" if count > grid.max() / 2 else "black")
    plt_save_and_close(fig, str(output / "pathway_audit"))

    eligible = [row for row in rows if (example is None or row.name == example) and row.rho_edge is not None]
    if example is not None and not eligible:
        raise ValueError(f"no revised concordance trajectory for {example}")
    if eligible:
        row = eligible[0]
        fig, ax = plt_figure(total_width=7, hw_ratio=0.55)
        ax.plot(row.x_indices, row.rho_edge, label="Follow-edge endpoints")
        if row.rho_post is not None:
            # The first state is initialization, before any posts have been shown.
            ax.plot(row.x_indices, row.rho_post, label="Followee posts shown during the update")
        ax.axhline(uniform_concordance_probability(row.tolerance), ls="--", color="0.5", label="Uniform reference")
        ax.set(xlabel="Completed step", ylabel="Agent-weighted concordance", title=row.name, ylim=(0, 1.02))
        ax.legend()
        plt_save_and_close(fig, str(output / "edge_and_post_concordance"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=statistics_database())
    parser.add_argument(
        "--legacy", type=Path, action="append", help="Archived statistics database. Repeat to add files."
    )
    parser.add_argument("--origin", help="Only compare records from this workspace origin.")
    parser.add_argument("--example", help="Scenario name for the edge and post concordance plot.")
    parser.add_argument("--output", type=Path, default=PATHWAY_OUTPUT.resolve() / "revision")
    args = parser.parse_args(argv)
    legacy = args.legacy or [STAT_INPUT.resolve() / name for name in ("stats.db", "stats.eps.db")]
    analyze(args.database, legacy, args.output, args.origin, args.example)


if __name__ == "__main__":
    main()
