"""Recalculate the descriptive group contrasts reported in the original paper."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common.io.json import atomic_json
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import read_statistics, recorded_endpoint_complement
from experiments.paper.paths import PATHWAY_OUTPUT, statistics_database
from experiments.paper.pathway_index.data_utils import piecewise_linear_integral_trapz

METRICS = ("rewiring_events", "normalized_rewiring_time", "subjective_index_auc", "closed_triads")
HISTORY_SCOPES = ("zero", "original-design")


def history_in_scope(policy: str, history: int | None, scope: str) -> bool:
    """Select the declared design set, without inventing structural history."""
    if scope not in HISTORY_SCOPES:
        raise ValueError(f"unknown history scope: {scope}")
    if policy == "StructureM9":
        return history == 0
    if policy == "OpinionM9":
        return history == 0 if scope == "zero" else history in (0, 2, 6)
    return False


def _metrics(row: ScenarioStatistics) -> dict[str, float | None]:
    timing = auc = None
    if row.active_step is not None and row.active_step > 0:
        if row.event_step_mean is not None:
            timing = row.event_step_mean / row.active_step
        if row.x_indices is not None and row.g_index is not None:
            auc = float(piecewise_linear_integral_trapz(row.x_indices / row.active_step, row.g_index, 0, 1))
    return {
        "rewiring_events": row.event_count,
        "normalized_rewiring_time": timing,
        "subjective_index_auc": auc,
        "closed_triads": row.triads,
    }


def summarize(rows: list[ScenarioStatistics], cutoff: float, *, history_scope="zero") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize an explicit history design and descriptive I_w split.

    Contrasts compare group means. They are not paired intervention effects;
    percentages use the named reference group's mean as the denominator.
    ``original-design`` pools OpinionM9 histories 0/2/6 against StructureM9 0;
    ``zero`` is the separate zero-history restriction retained by default.
    """

    if not math.isfinite(cutoff):
        raise ValueError("cutoff must be finite")
    if history_scope not in HISTORY_SCOPES:
        raise ValueError(f"unknown history scope: {history_scope}")
    observations = []
    for row in rows:
        if not row.name.startswith("s_grad"):
            continue
        if not history_in_scope(row.recsys_type, row.post_retain_count, history_scope):
            continue
        if row.pathway_integral is None or not math.isfinite(row.pathway_integral):
            continue
        observations.append(
            {
                "name": row.name,
                "origin": row.origin,
                "analysis_id": row.analysis_id,
                "endpoint_complement": recorded_endpoint_complement(row),
                "group": "low" if row.pathway_integral < cutoff else "high",
                "policy": row.recsys_type,
                "reposting": "off" if row.repost_rate == 0 else "on",
                "I_w": row.pathway_integral,
                **_metrics(row),
            }
        )
    if not observations:
        if history_scope == "zero":
            raise ValueError("no finite zero-history gradation records for the matched comparison")
        raise ValueError("no finite gradation records for the selected history scope")
    data = pd.DataFrame(observations)
    summaries = []
    for dimension in ("policy", "reposting"):
        for (group, category), block in data.groupby(["group", dimension], sort=True):
            for metric in METRICS:
                values = pd.to_numeric(block[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                summaries.append(
                    {
                        "dimension": dimension,
                        "category": category,
                        "group": group,
                        "metric": metric,
                        "n_runs": len(block),
                        "n_observed": len(values),
                        "mean": float(values.mean()) if len(values) else None,
                        "population_std": float(values.std()) if len(values) else None,
                    }
                )
    summary = pd.DataFrame(summaries)
    means = {(r["dimension"], r["category"], r["group"], r["metric"]): r["mean"] for r in summaries}
    contrasts = []

    def contrast(kind, metric, target, reference, *, category=None, group=None):
        target_mean, reference_mean = means.get(target), means.get(reference)
        valid = target_mean is not None and reference_mean is not None
        difference = target_mean - reference_mean if valid else None
        contrasts.append(
            {
                "comparison": kind,
                "category": category,
                "group": group,
                "metric": metric,
                "target_mean": target_mean,
                "reference_mean": reference_mean,
                "difference": difference,
                "relative_percent": 100 * difference / reference_mean if valid and reference_mean != 0 else None,
            }
        )

    for metric in METRICS:
        for dimension, categories in (("policy", ("OpinionM9", "StructureM9")), ("reposting", ("off", "on"))):
            for category in categories:
                contrast(
                    "high_minus_low",
                    metric,
                    (dimension, category, "high", metric),
                    (dimension, category, "low", metric),
                    category=f"{dimension}:{category}",
                )
        for group in ("low", "high"):
            contrast(
                "opinion_minus_structure",
                metric,
                ("policy", "OpinionM9", group, metric),
                ("policy", "StructureM9", group, metric),
                group=group,
            )
            contrast(
                "reposting_on_minus_off",
                metric,
                ("reposting", "on", group, metric),
                ("reposting", "off", group, metric),
                group=group,
            )
    return data, summary, pd.DataFrame(contrasts)


def analyze(database, output, *, cutoff, origin=None, analysis_id=None, history_scope="zero"):
    rows = read_statistics(database, mode="gradation", origin=origin, analysis_id=analysis_id)
    data, summary, contrasts = summarize(rows, cutoff, history_scope=history_scope)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "grouped_run_metrics.csv", index=False)
    summary.to_csv(output / "grouped_metric_summary.csv", index=False)
    contrasts.to_csv(output / "grouped_metric_contrasts.csv", index=False)
    atomic_json(
        output / "grouped_summary_receipt.json",
        {
            "schema": 1,
            "analysis_id": rows[0].analysis_id,
            "origin": rows[0].origin,
            "endpoint_complement": recorded_endpoint_complement(rows[0]),
            "cutoff": cutoff,
            "grouping": "descriptive low/high I_w; not first-passage labels",
            "history_scope": history_scope,
            **({"history": 0} if history_scope == "zero" else {}),
            "policy_histories": {"StructureM9": [0], "OpinionM9": [0] if history_scope == "zero" else [0, 2, 6]},
            "policies": ["OpinionM9", "StructureM9"],
            "runs": len(data),
            "metrics": list(METRICS),
            "relative_percent": "100 * (target_mean - reference_mean) / reference_mean; missing when reference is zero",
        },
    )
    return {"runs": len(data), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=statistics_database())
    parser.add_argument("--output", type=Path, default=PATHWAY_OUTPUT.resolve() / "revision")
    parser.add_argument(
        "--cutoff", type=float, required=True, help="Explicit descriptive split; historical value was 0.6."
    )
    parser.add_argument("--origin")
    parser.add_argument("--analysis-id")
    parser.add_argument("--history-scope", choices=HISTORY_SCOPES, default="zero")
    args = parser.parse_args(argv)
    return analyze(args.database, args.output, cutoff=args.cutoff, origin=args.origin,
                   analysis_id=args.analysis_id, history_scope=args.history_scope)


if __name__ == "__main__":
    main()
