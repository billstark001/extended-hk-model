"""Read a single, explicit statistics revision for the original-paper figures."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from pathlib import Path

from common.settings import environment_value
from experiments.paper.analysis.records import ScenarioStatistics, StatisticsStore
from experiments.paper.paths import statistics_database

StatisticFilter = Callable[[ScenarioStatistics], bool]


def read_statistics(
    database: str | Path | None = None, *, mode: str, origin: str | None = None, analysis_id: str | None = None
) -> list[ScenarioStatistics]:
    """Load one mode without pooling incompatible revisions or duplicate raw runs.

    ``EHK_STATS_ORIGIN`` and ``EHK_ANALYSIS_ID`` select an explicit subset when a
    document store contains multiple source datasets or analysis protocols.
    Reading closes the SQLite connection before plotting begins.
    """

    origin = origin or environment_value("EHK_STATS_ORIGIN")
    analysis_id = analysis_id or environment_value("EHK_ANALYSIS_ID")
    with StatisticsStore(database or statistics_database(), readonly=True) as store:
        rows = store.records(origin=origin, mode=mode, analysis_id=analysis_id)
    if not rows:
        raise ValueError(f"no {mode!r} statistics match the selected dataset and analysis revision")
    revisions = {row.analysis_id for row in rows}
    if len(revisions) != 1:
        raise ValueError("multiple analysis revisions match; select EHK_ANALYSIS_ID before plotting")
    if len({row.origin for row in rows}) != 1:
        raise ValueError("multiple source datasets match; select EHK_STATS_ORIGIN before plotting")
    stale = [row.name for row in rows if getattr(row, "stale", False) or getattr(row, "complete", True) is False]
    if stale:
        raise ValueError(f"stale or incomplete statistics must be reanalyzed before plotting: {', '.join(stale[:5])}")
    identities = [(row.name, row.origin) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate source identities in statistics selection")
    return rows


def select_statistics(records: Iterable[ScenarioStatistics], *predicates: StatisticFilter) -> list[ScenarioStatistics]:
    """Filter decoded documents using ordinary Python predicates."""

    return [row for row in records if all(predicate(row) for predicate in predicates)]


def descriptive_iw_cutoff() -> float:
    """Require an explicit cutoff when reproducing the historical grouped plots.

    The historical value was 0.6. Its use with revised indices is an analysis
    choice, not an established first-passage classifier.
    """

    raw = environment_value("EHK_IW_CUTOFF")
    if raw is None:
        raise ValueError("set EHK_IW_CUTOFF explicitly for grouped I_w plots; the historical cutoff was 0.6")
    cutoff = float(raw)
    if not math.isfinite(cutoff):
        raise ValueError("EHK_IW_CUTOFF must be finite")
    return cutoff


def is_low_iw(row: ScenarioStatistics) -> bool:
    """Select the low-integral group for an explicitly chosen descriptive split."""

    cutoff = descriptive_iw_cutoff()
    return row.pathway_integral is not None and math.isfinite(row.pathway_integral) and row.pathway_integral < cutoff


def is_high_iw(row: ScenarioStatistics) -> bool:
    """Select the complementary high-integral group, retaining finite values only."""

    cutoff = descriptive_iw_cutoff()
    return row.pathway_integral is not None and math.isfinite(row.pathway_integral) and row.pathway_integral >= cutoff
