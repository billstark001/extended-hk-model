"""Attribute archived/new Figure 5 differences using existing saved statistics.

Both inputs are read-only. This audit never runs a model or regenerates a state
metric. It separates history pooling, batch coverage, I_w grouping, activity
clocks and saved I_s curves. SQLite remains the canonical derived result.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from common.storage.documents import DocumentStore, DocumentWrite
from ehk.metrics.trajectory import pathway_index
from experiments.paper.pathway_index.data_utils import piecewise_linear_integral_trapz

ROOT = Path(__file__).resolve().parents[3]
PRIMARY = "e499b1ab5ff7ddc6480c5578127996e1945eecfd78af2f9a605f0fb7b93a267e"
ORIGINS = {"data2": "original-data2", "ssd_tmp": "original-data3"}
TABLES = "ehk-legacy-stats-audit-tables-v1"
REPORTS = "ehk-legacy-stats-audits-v1"
HISTORICAL = "4e7a9d2"
METRICS = ("rewiring_events", "normalized_rewiring_time", "subjective_index_auc", "closed_triads")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def historical_auc():
    source = git("show", f"{HISTORICAL}:works/plot/data_utils.py")
    tree = ast.parse(source.decode("utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "piecewise_linear_integral_trapz")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "historical_auc", "exec"), namespace)
    return namespace[function.name], hashlib.sha256(source).hexdigest()


def read_legacy(database):
    columns = ("name,origin,tolerance,decay,rewiring,retweet,recsys_type,tweet_retain_count,"
               "step,active_step,grad_index,event_count,event_step_mean,triads,last_opinion_peak_count,"
               "x_indices,h_index,p_index,g_index")
    result = []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        for saved in connection.execute(f"SELECT {columns} FROM scenariostatistics ORDER BY origin,name"):
            row = dict(saved)
            for field in ("x_indices", "h_index", "p_index", "g_index"):
                row[field] = np.load(io.BytesIO(row[field]), allow_pickle=False)
            row.update(policy=row["recsys_type"], kh=int(row["tweet_retain_count"]),
                       p=row["retweet"], Iw=row["grad_index"])
            result.append(row)
    require(len(result) == 36854, "Archived cohort size changed")
    require(len({(r['origin'], r['name']) for r in result}) == len(result), "Duplicate archived identity")
    return result


def read_primary(database):
    fields = ("name", "origin", "tolerance", "decay", "rewiring", "step", "active_step",
              "event_count", "event_step_mean", "triads", "last_opinion_peak_count",
              "x_indices", "h_index", "p_index", "g_index")
    rows = []
    with DocumentStore(database, readonly=True) as store:
        store.connection.execute("BEGIN")
        for doc in store.values("ehk-paper-statistics-v3"):
            if doc["analysis_id"] != PRIMARY:
                continue
            require(doc["complete"] and not doc["stale"] and doc["metric_revision"] == 6
                    and doc["endpoint_complement"], "Unexpected primary convention")
            row = {name: doc[name] for name in fields}
            row.update(policy=doc["recsys_type"], kh=doc["post_retain_count"],
                       p=doc["repost_rate"], Iw=doc["pathway_integral"])
            rows.append(row)
    require(len(rows) == 24566, "Primary cohort size changed")
    require(set(r["origin"] for r in rows) == set(ORIGINS.values()), "Unexpected primary origins")
    return rows


def finite_iw(row):
    return row["Iw"] is not None and math.isfinite(row["Iw"])


def metric_values(row, auc, *, clock=None):
    clock = row["active_step"] if clock is None else clock
    require(clock is not None and clock > 0, "Nonpositive activity time")
    x, g = row["x_indices"], row["g_index"]
    require(x[0] == 0 and x[-1] == row["step"] and clock <= x[-1], "Uncovered activity interval")
    return (float(row["event_count"]), float(row["event_step_mean"] / clock),
            float(auc(x / clock, g, 0, 1)), float(row["triads"]))


def summarize(stage, rows, labels, values):
    summaries, contrasts, means = [], [], {}
    categories = {"St": lambda r: r["policy"] == "StructureM9",
                  "Op": lambda r: r["policy"] == "OpinionM9",
                  "!R": lambda r: r["p"] == 0, "R": lambda r: r["p"] > 0}
    for category, predicate in categories.items():
        for group in ("low", "high"):
            chosen = [i for i, row in enumerate(rows) if predicate(row)
                      and (labels[i] >= .6) == (group == "high")]
            block = np.asarray([values[i] for i in chosen])
            require(len(block) > 0 and np.isfinite(block).all(), "Empty/nonfinite group")
            for j, metric in enumerate(METRICS):
                mean = float(block[:, j].mean())
                means[category, group, metric] = mean
                summaries.append(dict(stage=stage, category=category, group=group, metric=metric,
                                      n=len(block), mean=mean, population_std=float(block[:, j].std())))
    for metric in METRICS:
        pairs = [("high_minus_low", category, "", (category, "high"), (category, "low"))
                 for category in categories]
        pairs += [(comparison, "", group, (target, group), (reference, group))
                  for group in ("low", "high")
                  for comparison, target, reference in (("opinion_minus_structure", "Op", "St"),
                                                        ("reposting_on_minus_off", "R", "!R"))]
        for comparison, category, group, target, reference in pairs:
            a, b = means[*target, metric], means[*reference, metric]
            contrasts.append(dict(stage=stage, comparison=comparison, category=category, group=group,
                                  metric=metric, target_mean=a, reference_mean=b,
                                  relative_percent=100 * (a - b) / b))
    return summaries, contrasts


def baseline_interval(p, h, epsilon):
    """Conservative I_w bounds after changing only the old I_h baseline.

    Positive old h recovers the raw ratio. A clipped zero loses that ratio,
    so its corrected value ranges from zero to a. Signed trapezoidal weights
    propagate this missing information without inventing raw observations.
    """
    p, h = np.asarray(p, dtype=float), np.asarray(h, dtype=float)
    require(p.shape == h.shape and p.ndim == 1 and len(p) > 0, "Invalid archived arrays")
    require(np.isfinite(p).all() and np.isfinite(h).all() and np.all(h >= 0), "Invalid archived values")
    old_b, new_b = epsilon - epsilon**2 / 8, epsilon - epsilon**2 / 4
    a = (old_b - new_b) / (1 - new_b)
    upper_h = a + (1 - a) * h
    lower_h = np.where(h > 0, upper_h, 0.)
    weights = np.zeros(len(p))
    if len(p) > 1:
        change = np.diff(p) / 2
        weights[:-1] += change
        weights[1:] += change
    weights[-1] += 1 - p[-1]
    lower = np.where(weights >= 0, lower_h, upper_h)
    upper = np.where(weights >= 0, upper_h, lower_h)
    return float(weights @ lower), float(weights @ upper)


def optional_mean_bounds(fixed, optional):
    require(len(fixed) > 0, "No mandatory high runs")
    count, total = len(fixed), sum(fixed)
    low = (total + np.r_[0., np.cumsum(sorted(optional))]) / (count + np.arange(len(optional) + 1))
    high = (total + np.r_[0., np.cumsum(sorted(optional, reverse=True))]) / (count + np.arange(len(optional) + 1))
    return float(low.min()), float(high.max())


def analyze(legacy_database, database, output):
    legacy_database, database, output = map(lambda p: Path(p).resolve(), (legacy_database, database, output))
    require(output / "analysis.sqlite" not in (legacy_database, database), "Output overlaps source")
    before = {str(p): sha(p) for p in (legacy_database, database)}
    old, new = read_legacy(legacy_database), read_primary(database)
    old_index = {(r["origin"], r["name"]): r for r in old}
    pairs = [(old_index[next(k for k, v in ORIGINS.items() if v == r["origin"]), r["name"]], r) for r in new]
    invariants = ("name", "tolerance", "decay", "rewiring", "policy", "kh", "p", "step",
                  "event_count", "event_step_mean", "triads", "last_opinion_peak_count")
    require(all(all(a[k] == b[k] for k in invariants) for a, b in pairs), "Trajectory invariant mismatch")
    require(all({r['name'] for r in old if r['origin'] == a} ==
                {r['name'] for r in new if r['origin'] == b} for a, b in ORIGINS.items()), "Coverage mismatch")
    auc_old, auc_source_sha = historical_auc()
    old_values = {id(r): metric_values(r, auc_old) for r in old}
    old_valid = [r for r in old if finite_iw(r)]
    max_old_iw_residual = max(abs(pathway_index(r['p_index'], r['h_index']) - r['Iw']) for r in old_valid)
    require(max_old_iw_residual < 1e-10, "Archived I_w differs from signed completed integral")
    common = [(a, b) for a, b in pairs if a["kh"] == 0 and finite_iw(a)]
    primary_kh0 = [r for r in new if r["kh"] == 0]
    stages = []
    def add(stage, description, rows, labels, values):
        stages.append((stage, description, rows, labels, values))
    def old_stage(stage, description, predicate):
        rows = [r for r in old_valid if predicate(r)]
        add(stage, description, rows, [r["Iw"] for r in rows], [old_values[id(r)] for r in rows])
    old_stage("A", "Archived three batches; all histories", lambda r: True)
    old_stage("B", "Same archived batches; kh=0", lambda r: r["kh"] == 0)
    old_stage("C", "Shared two batches; kh=0; archived metrics", lambda r: r["kh"] == 0 and r["origin"] in ORIGINS)
    old_stage("C_pool", "Shared two batches; all archived histories", lambda r: r["origin"] in ORIGINS)
    rows = [a for a, b in common]
    labels = [b["Iw"] for a, b in common]
    add("D", "Same C identities; new Iw labels; archived outcomes/clock", rows, labels, [old_values[id(a)] for a, b in common])
    add("E", "Same D identities/labels; new activity clock; archived Is", rows, labels,
        [metric_values(a, auc_old, clock=b['active_step']) for a, b in common])
    add("F", "Same identities/labels; new activity clock and Is", rows, labels,
        [metric_values(b, piecewise_linear_integral_trapz) for a, b in common])
    add("G", "Current full two-batch kh=0 cohort; include previously NULL Iw", primary_kh0,
        [r["Iw"] for r in primary_kh0], [metric_values(r, piecewise_linear_integral_trapz) for r in primary_kh0])
    add("G_pool", "Current two batches; all histories; retain historical comparison scope", new,
        [r["Iw"] for r in new], [metric_values(r, piecewise_linear_integral_trapz) for r in new])
    tables = {"grouped_summary": [], "contrasts": [], "cohort_steps": [], "identity": [],
              "history_composition": [], "migration": [], "baseline_bounds": [], "scalar_changes": []}
    for stage, description, rows, labels, values in stages:
        summaries, contrasts = summarize(stage, rows, labels, values)
        tables["grouped_summary"].extend(summaries); tables["contrasts"].extend(contrasts)
        pick = lambda category: next(r for r in summaries if r['category'] == category and r['group'] == 'high' and r['metric'] == 'closed_triads')
        op, st = pick('Op'), pick('St')
        tables["cohort_steps"].append(dict(stage=stage, description=description, n=len(rows),
            high_Op_n=op['n'], high_St_n=st['n'], high_Op_mean_triads=op['mean'],
            high_St_mean_triads=st['mean'], Op_vs_St_high_triads_percent=100*(op['mean']/st['mean']-1)))
    for origin, current in ORIGINS.items():
        chosen = [(a, b) for a, b in pairs if a['origin'] == origin]
        tables['identity'].append(dict(legacy_origin=origin, primary_origin=current, n=len(chosen),
            exact_invariants=len(chosen), active_step_changed=sum(a['active_step'] != b['active_step'] for a, b in chosen)))
    for policy, kh, group in sorted({(r['policy'], r['kh'], 'high' if r['Iw'] >= .6 else 'low') for r in old_valid}):
        selected = [r for r in old_valid if (r['policy'], r['kh'], 'high' if r['Iw'] >= .6 else 'low') == (policy, kh, group)]
        tables['history_composition'].append(dict(policy=policy, kh=kh, group=group, n=len(selected), mean_triads=float(np.mean([r['triads'] for r in selected]))))
    for kh in ("all", 0):
        chosen = [(a, b) for a, b in pairs if finite_iw(a) and (kh == 'all' or a['kh'] == kh)]
        for policy in ("StructureM9", "OpinionM9"):
            for old_group, new_group in itertools.product(("low", "high"), repeat=2):
                block = [(a, b) for a, b in chosen if a['policy'] == policy and
                         ('high' if a['Iw'] >= .6 else 'low') == old_group and ('high' if b['Iw'] >= .6 else 'low') == new_group]
                tables['migration'].append(dict(kh=str(kh), policy=policy, old_group=old_group, new_group=new_group,
                    n=len(block), mean_triads=float(np.mean([a['triads'] for a, b in block])) if block else None))
    for scope, chosen in (("all_matched", pairs), ("kh0_matched", [(a, b) for a, b in pairs if a['kh'] == 0])):
        for field in ('active_step', 'Iw'):
            valid = [(a, b) for a, b in chosen if a[field] is not None and b[field] is not None]
            delta = np.array([b[field] - a[field] for a, b in valid])
            tables['scalar_changes'].append(dict(scope=scope, field=field, n=len(valid), changed=int(np.count_nonzero(delta)),
                old_median=float(np.median([a[field] for a, b in valid])), new_median=float(np.median([b[field] for a, b in valid])),
                median_delta=float(np.median(delta)), q05_delta=float(np.quantile(delta,.05)), q95_delta=float(np.quantile(delta,.95))))
    bounds = {}
    for policy in ('StructureM9', 'OpinionM9'):
        selected = [a for a, b in common if a['policy'] == policy]
        intervals = [(r, baseline_interval(r['p_index'], r['h_index'], r['tolerance'])) for r in selected]
        fixed = [r['triads'] for r,(lo,hi) in intervals if lo >= .6]
        optional = [r['triads'] for r,(lo,hi) in intervals if lo < .6 <= hi]
        mean_lo, mean_hi = optional_mean_bounds(fixed, optional)
        bounds[policy] = mean_lo, mean_hi
        tables['baseline_bounds'].append(dict(policy=policy, n=len(selected), old_high_n=sum(r['Iw'] >= .6 for r in selected),
            mandatory_high_n=len(fixed), ambiguous_high_n=len(optional), mean_high_triads_lower=mean_lo, mean_high_triads_upper=mean_hi))
    baseline_contrast = [100*(bounds['OpinionM9'][0]/bounds['StructureM9'][1]-1),
                         100*(bounds['OpinionM9'][1]/bounds['StructureM9'][0]-1)]
    require(all(sha(Path(p)) == expected for p, expected in before.items()), "Input database changed during audit")
    receipt = dict(schema=1, legacy_database=str(legacy_database), primary_database=str(database),
        input_sha256=before, primary_analysis_id=PRIMARY, legacy_rows=len(old), primary_rows=len(new),
        legacy_null_Iw=sum(not finite_iw(r) for r in old), matched_kh0_old_finite=len(common),
        historical_plot_commit=git('rev-parse', HISTORICAL).decode().strip(), historical_auc_source_sha256=auc_source_sha,
        historical_plot_sha256=hashlib.sha256(git('show',f'{HISTORICAL}:works/plot/f_grad_index_interpret.py')).hexdigest(),
        script_sha256=sha(__file__), archived_Iw_max_integration_residual=max_old_iw_residual,
        analysis_source_sha256={name: sha(ROOT/name) for name in (
            'src/ehk/metrics/trajectory.py', 'src/common/numerics/arrays.py',
            'src/common/storage/documents.py', 'experiments/paper/pathway_index/data_utils.py')},
        python_version=sys.version.split()[0], numpy_version=np.__version__,
        baseline_only_high_Op_vs_St_triads_percent_bounds=baseline_contrast,
        interpretation='Sequential descriptive bridge, order-dependent; not causal contribution shares. Baseline bounds retain clipping uncertainty. Archived DB has no frozen execution-source or raw hashes.')
    report_id = hashlib.sha256(canonical(receipt).encode()).hexdigest()
    receipt['report_id'] = report_id
    receipt['tables'] = {name: report_id+'/'+name for name in tables}
    output.mkdir(parents=True, exist_ok=True)
    with DocumentStore(output/'analysis.sqlite') as store:
        store.put_many([DocumentWrite(TABLES, receipt['tables'][name], rows) for name, rows in tables.items()]
                       + [DocumentWrite(REPORTS, report_id, receipt)])
        store.seal()
    return receipt


def export(database, report_id, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    tables = {}
    with DocumentStore(database, readonly=True) as store:
        receipt = store.get(REPORTS, report_id)
        require(receipt is not None, "Unknown audit report")
        for name, key in receipt['tables'].items():
            rows = store.get(TABLES, key)
            tables[name] = rows
            with (output/(name+'.csv')).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'provenance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plot_exhibits(tables, output)


def plot_exhibits(tables, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from common.plotting import setup_paper_params

    setup_paper_params()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout='constrained')
    history = sorted((r for r in tables['history_composition']
                      if r['policy'] == 'OpinionM9' and r['group'] == 'high'), key=lambda r: r['kh'])
    st = next(r for r in tables['history_composition'] if r['policy'] == 'StructureM9' and r['group'] == 'high')
    means = [r['mean_triads'] for r in history]
    axes[0].bar(range(3), means, color='tab:blue', alpha=.65)
    axes[0].axhline(st['mean_triads'], color='.25', ls='--', label='Structure, $k_h=0$')
    for i, row in enumerate(history):
        axes[0].text(i, row['mean_triads']+500, f"n={row['n']}", ha='center', fontsize=8)
    axes[0].set(xticks=range(3), xticklabels=[r['kh'] for r in history],
                xlabel='Opinion history $k_h$', ylabel='Closed triads', ylim=(0, 26000))
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    axes[0].set_title('(a)', loc='left')
    selected = [next(r for r in tables['cohort_steps'] if r['stage'] == s) for s in ('A','B','C','D','G')]
    values = [r['Op_vs_St_high_triads_percent'] for r in selected]
    axes[1].bar(range(5), values, color='tab:blue', alpha=.65)
    axes[1].axhline(0, color='.25', lw=1)
    for i, value in enumerate(values):
        axes[1].text(i, value+(0.5 if value>0 else -0.7), f'{value:+.2f}',
                     ha='center', va='bottom' if value>0 else 'top', fontsize=8)
    axes[1].set(xticks=range(5), xticklabels=['Original\npool','Match\nhistory','Shared\nbatches','Update\n$I_w$','Current'],
                ylabel='High-group Op versus St (%)', ylim=(-19, 11))
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].set_title('(b)', loc='left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)
    for suffix in ('pdf','png'):
        fig.savefig(output/f'legacy_difference_bridge.{suffix}',dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('analyze')
    build.add_argument('--legacy-database', type=Path, required=True)
    build.add_argument('--database', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--export-directory', type=Path)
    render = commands.add_parser('export')
    render.add_argument('--database', type=Path, required=True)
    render.add_argument('--report-id', required=True)
    render.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'analyze':
        receipt = analyze(args.legacy_database, args.database, args.output)
        if args.export_directory:
            export(args.output/'analysis.sqlite', receipt['report_id'], args.export_directory)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    else:
        export(args.database,args.report_id,args.output)


if __name__ == '__main__':
    main()
