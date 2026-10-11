"""Offline analysis dispatch; no date-specific scripts or solver imports."""

import json
from pathlib import Path

from common.io.json import atomic_json, sha
from common.io.tables import table
from theory.runtime.dataset import checked_records, validate

from .kernel_validation.analyze.cases import analyze as analyze_kernel
from .micro_validation.analyze.cases import analyze_micro
from .micro_validation.analyze.ensembles import analyze_ensembles as micro_ensembles
from .operator_validation.analyze.cases import analyze_operator
from .release_clocks.analyze.cases import analyze_clocks
from .release_clocks.analyze.ensembles import analyze_ensembles as clock_ensembles
from .runtime.comparison import baseline_prefixes
from .runtime.paths import OutputPaths


def analyze_run(output=None, label="validation", families=None, baseline_label=None):
    """Validate a run binding, then write study tables/figures from completed raw records.

    families selects analysis families without changing the frozen plan. Ensemble
    methods use its saved analysis_protocol; baseline_label optionally checks
    matching prefixes. Return the summary including completed and missing counts.
    No solver is imported or run.
    """
    outputs = OutputPaths(output)
    root = outputs.control(label)
    plan, binding, expected = validate(root)
    families = list(dict.fromkeys(j["family"] for j in plan["jobs"])) if families is None else families
    if any(f not in ("micro", "clocks", "operator", "kernel") for f in families):
        raise ValueError("unknown analysis family")
    protocol = plan.get("analysis_protocol")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "binding": binding,
        "smoke_only": label.startswith("smoke_"),
        "expected": len(expected),
        "analysis_protocol": protocol,
        "families": {},
    }
    completed = 0
    for family in families:
        items = checked_records(outputs, label, family, binding, expected)
        completed += len(items)
        destination = out / family if protocol else outputs.family(family, label) / "analysis"
        destination.mkdir(parents=True, exist_ok=True)
        if family == "kernel":
            rows, curves = [], []
            for record in items:
                r, c = analyze_kernel(record)
                rows.extend(r)
                curves.extend(c)
            table(destination / "finite_list_audit.csv", rows)
            atomic_json(destination / "jump_distributions.json", curves)
            if any(r.get("mean_error", 0) > 1e-10 or r.get("second_error", 0) > 1e-10 for r in rows):
                raise ValueError("exact finite-list moments disagree with exported closure")
            result = {"rows": len(rows)}
        elif family == "micro":
            result = micro_ensembles(items, protocol, destination) if protocol else analyze_micro(items, destination)
        elif family == "clocks":
            result = (
                clock_ensembles(items, [j for j in plan["jobs"] if j["family"] == family], protocol, destination)
                if protocol
                else analyze_clocks(items, destination)
            )
            if protocol and baseline_label:
                summary["baseline"] = baseline_prefixes(items, outputs.control(baseline_label), binding, out)
        else:
            result = analyze_operator(items, destination)
        summary["families"][family] = result
        summary[family] = result
    summary.update(
        complete=completed,
        missing=len(expected) - completed,
        analysis_sources={
            str(p.relative_to(Path(__file__).parent)): sha(p)
            for p in Path(__file__).parent.rglob("*.py")
            if "analyze" in p.parts or p.name == "analysis.py"
        },
    )
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary
