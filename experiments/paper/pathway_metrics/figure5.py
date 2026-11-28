"""Reproduce Figure 5 at I_w cutoffs 0.6 and 0.7 from the formal SQLite cohort.

Run as a model-repository module with an explicit output directory. Inputs are
read-only; CSVs are compact generated exhibits, never analysis input caches.
No raw trajectory, model execution, state metric or I_w is recalculated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ANALYSIS = "e499b1ab5ff7ddc6480c5578127996e1945eecfd78af2f9a605f0fb7b93a267e"
NAMESPACE = "ehk-paper-statistics-v3"
MODEL_ROOT = Path(__file__).resolve().parents[3]
CUTOFFS = (.6, .7)
METRICS = ("rewiring_events", "normalized_rewiring_time", "subjective_index_auc", "closed_triads")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_composition(data, cutoff):
    rows = []
    for dimension, categories in (("policy", ("StructureM9", "OpinionM9")), ("reposting", ("off", "on"))):
        for category in categories:
            for group in ("low", "high"):
                block = data[(data[dimension] == category) & (data["group"] == group)]
                for axis, levels in (("origin", ("original-data2", "original-data3")),
                                     ("p", (0., .1, .25, .5)),
                                     ("D_sign", ("q<alpha", "q=alpha", "q>alpha")),
                                     ("kh", (0, 2, 6))):
                    for level in levels:
                        count = int((block[axis] == level).sum())
                        rows.append(dict(cutoff=cutoff, dimension=dimension, category=category, group=group,
                                         group_n=len(block), axis=axis, level=level, n=count,
                                         fraction=count/len(block) if len(block) else None))
    return rows


def history_statistics(data, cutoff):
    """Aggregate the same evaluated metrics by each declared history condition."""
    rows = []
    totals = data.groupby(["policy", "group"]).size()
    for (policy, kh, group), block in data.groupby(["policy", "kh", "group"], sort=True):
        total = int(totals.loc[(policy, group)])
        for metric in METRICS:
            values = block[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(dict(cutoff=cutoff, policy=policy, kh=int(kh), group=group,
                             metric=metric, n_runs=len(block), n_observed=len(values),
                             mean=float(values.mean()) if len(values) else None,
                             population_std=float(values.std()) if len(values) else None,
                             policy_group_n=total, history_weight=len(block)/total))
    return rows


def plot(summary, cutoff, limits, destination):
    from common.plotting import plt_save_and_close
    from experiments.paper.pathway_index.bar_plot import render_figure5

    fig, _ = render_figure5(summary, limits=limits)
    plt_save_and_close(fig, str(destination / f"figure5_iw_{cutoff:.1f}"))


def write_report(destination, history_scope, database, records, counts, by_cut, changes, gaps, compositions, histories):
    """Describe the selected design using its actual counts and calculated values."""
    original = history_scope == "original-design"
    scope_name = "原设计集合汇总" if original else "零历史辅助限制"
    policy_counts = Counter(r.recsys_type for r in records)
    history_counts = Counter((r.recsys_type, int(r.post_retain_count)) for r in records)
    repost_counts = Counter("!R" if r.repost_rate == 0 else "R" for r in records)
    count_index = {(r["cutoff"], r["dimension"], r["category"], r["group"]): r["n"] for r in counts}
    history_rows = [r for r in histories if r["metric"] == "closed_triads"]
    def comparison(key, cutoff):
        return by_cut[cutoff].loc[key]
    def pct(key, cutoff):
        return float(comparison(key, cutoff)["relative_percent"])
    def pair(key):
        return " / ".join(f"{pct(key, c):+.2f}%" for c in CUTOFFS)
    command = (f"{MODEL_ROOT.as_posix()}/.venv/Scripts/python.exe -m experiments.paper.pathway_metrics.figure5 "
               f"--history-scope {history_scope} --database {database.resolve().as_posix()} --output {destination.resolve().as_posix()}")
    design_text = (
        "本范围按原设计汇总 StructureM9 的 `k_h=0` 与 OpinionM9 的 `k_h=0,2,6`。Opinion 帖子的历史选择已有自然规则；Structure 若追加历史，需要另选随机、衰减等额外机制，属于新的模型设定。因此这里比较的是原设计的算法机制集合，不需要为这次复核新增 Structure 历史运行。各历史条件的分层同时保留。"
        if original else
        "本范围同时限制两算法 `k_h=0`，作为单独的辅助分层。原设计主范围是 StructureM9 `k_h=0` 与 OpinionM9 `k_h=0,2,6` 的集合汇总；两者回答不同范围的问题。Opinion 历史已有自然选择规则，Structure 追加历史则需另定义随机、衰减等机制，本辅助限制不要求这种新设定。"
    )
    lines = [
        f"# Figure 5：{scope_name}，0.6 与 0.7 工作分界", "",
        f"日期：2026-11-28。分析身份 `{ANALYSIS}`，来源为 data2+data3；`history_scope={history_scope}`。本范围共 **{len(records):,}** 条完整运行（StructureM9 {policy_counts['StructureM9']:,}，OpinionM9 {policy_counts['OpinionM9']:,}）。data1 和独立的 P/E 诊断不参与这里的汇总。", "",
        design_text, "",
        "## 定义与复现", "",
        "保留主库 revision 6 的 L 指标与默认终点补项；低组为 `I_w<c`、高组为 `I_w>=c`。状态指标和 I_w 不重算，模拟也不重跑。直接复用 `pathway_index.grouped_summary.summarize(..., history_scope=...)` 的唯一指标计算核心。", "",
        "- 换边数、终态闭合三元组数直接读主库。归一化换边时刻为每条运行完整 `[0,T]` 内的 `event_step_mean/max(active_step,1)`，事件窗口未在活动时刻截断。",
        "- AUC 精确调用 `piecewise_linear_integral_trapz(x_indices/max(active_step,1), g_index, 0, 1)`，保持原 Figure 5 的折线插值、边界和归一化时钟；不用 `mean(g_index)` 或 `g_index_mean_active` 替代。当前所有活动时刻均为正。",
        "- 每条运行等权，柱为算术均值及总体 SD（ddof=0）；不是先对各历史条件取均值再等权。百分比为 `100*(target_mean-reference_mean)/reference_mean`。SD 不作置信区间或显著性检验。", "",
        "```powershell", f"Set-Location {MODEL_ROOT.as_posix()}", command, "```", "",
        f"`--output` 必填，输入路径可通过 `--database` 显式指定。当前 scope 核对 {len(records):,} 条样本；快照 SHA、协议和源码 SHA 记录在 `provenance.csv`。CSV 是展品，不作输入缓存，本目录不复制计算代码或逐运行数据。", "",
        "## 设计集合与权重", "",
        "| 算法 | kh | 完整运行数 | 在算法集合中的运行权重 |", "| --- | ---: | ---: | ---: |",
    ]
    for (policy, kh), n in sorted(history_counts.items()):
        lines.append(f"| {policy} | {kh} | {n:,} | {n/policy_counts[policy]:.4%} |")
    lines += ["", f"!R 仅 `p=0`，共 {repost_counts['!R']:,} 条；R 合并 `.1,.25,.5`，共 {repost_counts['R']:,} 条。高低组内的实际 kh 权重还会随分界改变：", "",
              "| 分界 | 算法 | kh | 高组 n | 高组内权重 | 三元组均值 |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in history_rows:
        if row["group"] == "high":
            lines.append(f"| {row['cutoff']:.1f} | {row['policy']} | {row['kh']} | {row['n_runs']:,} | {row['history_weight']:.4%} | {row['mean']:,.3f} |")
    lines += ["", "`history_groups.csv` 对每个算法／kh／高低组列出四项指标的均值、SD、有效 n 和历史权重；`group_composition.csv` 同时记录 kh、来源、p 和 D 的组成。原设计汇总与单独历史条件的分层相互补充。", "",
              "## 分界与样本迁移", "",
              "| 条件 | 0.6 低组 | 0.6 高组 | 0.7 低组 | 0.7 高组 | 从高组移入低组 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    moved = 0
    for dimension, category, label in (("policy", "StructureM9", "St"), ("policy", "OpinionM9", "Op"),
                                       ("reposting", "off", "!R"), ("reposting", "on", "R")):
        ns = [count_index[c, dimension, category, group] for c in CUTOFFS for group in ("low", "high")]
        if dimension == "policy": moved += ns[1]-ns[3]
        lines.append(f"| {label} | {ns[0]:,} | {ns[1]:,} | {ns[2]:,} | {ns[3]:,} | {ns[1]-ns[3]:,} |")
    lines += ["", f"共 {moved:,} 条（{moved/len(records):.2%}）位于 `[0.6,0.7)`，提高分界后从高组进入低组。分组在运行结束后形成，条件间参数和 kh 组成不同；以下是设计集合的描述性均值比较，不是同初态／同种子的配对干预效应。", "",
              "## 主要均值比较", ""]
    specs = (
        ("高组：R/!R 换边数", "+34%", ("reposting_on_minus_off", "", "high", "rewiring_events")),
        ("高组：Op/St 归一化换边时刻", "−32%", ("opinion_minus_structure", "", "high", "normalized_rewiring_time")),
        ("高组：R/!R 归一化换边时刻", "+41%", ("reposting_on_minus_off", "", "high", "normalized_rewiring_time")),
        ("St：高/低组三元组", "+17%", ("high_minus_low", "policy:StructureM9", "", "closed_triads")),
        ("Op：高/低组三元组", "+123%", ("high_minus_low", "policy:OpinionM9", "", "closed_triads")),
        ("!R：高/低组三元组", "+32%", ("high_minus_low", "reposting:off", "", "closed_triads")),
        ("R：高/低组三元组", "+98%", ("high_minus_low", "reposting:on", "", "closed_triads")),
        ("高组：Op/St 三元组", "+7%", ("opinion_minus_structure", "", "high", "closed_triads")),
    )
    lines += (["| 比较 | 旧正文约数 | 当前 0.6 | 当前 0.7 |", "| --- | ---: | ---: | ---: |"] if original else
              ["| 辅助范围比较 | 0.6 | 0.7 |", "| --- | ---: | ---: |"])
    for label, old, key in specs:
        values = [pct(key, c) for c in CUTOFFS]
        prefix = f"| {label} | {old} |" if original else f"| {label} |"
        lines.append(f"{prefix} {values[0]:+.4f}% | {values[1]:+.4f}% |")
    triad_key = ("opinion_minus_structure", "", "high", "closed_triads")
    vals = [comparison(triad_key, c) for c in CUTOFFS]
    lines += ["", "高组的三元组均值如下：", "", "| 分界 | St | Op | Op 相对 St |", "| ---: | ---: | ---: | ---: |"]
    for c, value in zip(CUTOFFS, vals):
        lines.append(f"| {c:.1f} | {value['reference_mean']:,.6f} | {value['target_mean']:,.6f} | {value['relative_percent']:+.6f}% |")
    if original:
        lines += ["", "这一原设计集合的高组 Op 三元组均值在两分界均小幅高于 St；0.7 的差值更接近零，不能据总体 SD 作显著性或机制效应判断。旧文约数的来源还包含旧三批数据与旧指标分组；范围和旧值的核对见独立 legacy-stats audit。", ""]
    else:
        lines += ["", "这里的数值只描述 kh=0 辅助限制。它与原设计的三历史集合不是同一个汇总对象，应与原设计范围的结果分别阅读。", ""]
    lines += [f"高组 Op/St 换边数差为 {pair(('opinion_minus_structure', '', 'high', 'rewiring_events'))}；换边时刻差为 {pair(('opinion_minus_structure', '', 'high', 'normalized_rewiring_time'))}；AUC 差为 {pair(('opinion_minus_structure', '', 'high', 'subjective_index_auc'))}。报告方向同时保留具体量级，分组份额和百分比都随分界而变。", "",
              "## 数值方向与解释边界", ""]
    reversed_rows = [r for r in changes if r["direction_reversed"]]
    lines.append(f"{len(changes)} 项标准比较中，0.6 与 0.7 间有 {len(reversed_rows)} 项百分比符号变化：")
    for row in reversed_rows:
        lines.append(f"- `{row['comparison']}`，`{row['category'] or row['group']}`，`{row['metric']}`：{row['relative_percent_at_0_6']:+.4f}% → {row['relative_percent_at_0_7']:+.4f}%。")
    if not reversed_rows: lines.append("- 无。")
    lines += ["", "接近零的数值符号或排序变化不作统计效应解读；标准比较详见 `threshold_changes.csv`。高低组差距大小的算法排序另列于 `group_gap_comparison.csv`。较大 AUC 仅描述该运行归一化活动时钟内的局部模板相似度，不自动代表物理时间更早。旧文低组“稳定”也需要明确指标与描述尺度，不能仅凭柱形相近断言。", "",
              "## 图形与文件", "",
              "八面板顺序保持原 Figure 5：上排按算法，下排按转发；列为换边数、归一化换边时刻、Is AUC、闭合三元组。沿用 common.plotting 字体和线宽，输出矢量 PDF、300-dpi PNG 及外置 caption。两阈值共用涵盖全部均值±SD端点的坐标范围；部分时刻 SD 下界为负时也保留完整对称离散度，不代表负的时刻观测。", "",
              "- `grouped_metric_summary.csv`、`grouped_metric_contrasts.csv`：柱与比较。",
              "- `history_groups.csv`、`group_composition.csv`：kh 分层及权重。",
              "- `group_counts.csv`、`threshold_changes.csv`、`group_gap_comparison.csv`：组数及阈值变化。",
              "- `provenance.csv`：输入快照、scope、协议和源码凭证。", "",
              "本图生成仅写 artifacts，不替换正文图5；SI 另说明设计范围。既有 kh=0 辅助目录保留不覆盖。", ""]
    (destination/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    readme = [f"# Figure 5：{scope_name}", "", design_text, "", "从模型仓库运行：", "", "```powershell",
              f"Set-Location {MODEL_ROOT.as_posix()}", command, "```", "",
              f"数据源：`{database.resolve().as_posix()}`；正式分析 `{ANALYSIS}`；data2+data3；本 scope n={len(records):,}。`--output` 必填。", "",
              "所有运行等权；Op 的历史条件按有效运行数汇总，各高低组实际权重见 `history_groups.csv`。!R 为 p=0，R 汇总三个正转发率。SD 是总体离散度，不是置信区间。两个 scope 共用同一指标核心，不新增状态指标或历史机制。", "",
              "- [中文分析](REPORT.md)", "- [0.6 PDF](figure5_iw_0.6.pdf) · [PNG](figure5_iw_0.6.png)",
              "- [0.7 PDF](figure5_iw_0.7.pdf) · [PNG](figure5_iw_0.7.png)",
              "- [各历史条件均值／SD／n／权重](history_groups.csv)", "- [全部柱](grouped_metric_summary.csv) · [全部比较](grouped_metric_contrasts.csv)",
              "- [组数](group_counts.csv) · [kh／p／D／来源组成](group_composition.csv)",
              "- [阈值变化](threshold_changes.csv) · [高低组差距排序](group_gap_comparison.csv) · [来源凭证](provenance.csv)", "",
              "本目录只保留说明、紧凑汇总和图形；代码位于模型仓库 `experiments/paper/pathway_metrics/figure5.py`。本图生成仅写 artifacts，不替换正文图5；SI 另说明设计范围。既有 kh=0 辅助展品保留不覆盖。", ""]
    (destination/"README.md").write_text("\n".join(readme), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite"))
    parser.add_argument("--output", type=Path, required=True, help="Explicit directory for Figure 5 exhibits; no source-tree default")
    parser.add_argument("--history-scope", choices=("zero", "original-design"), default="original-design")
    parser.add_argument("--expected-runs", type=int, help="Optional explicit override of the scope's frozen cohort check")
    args = parser.parse_args()
    from common.storage.documents import DocumentStore, decode_document
    from experiments.paper.analysis.records import ScenarioStatistics
    from experiments.paper.pathway_index.grouped_summary import history_in_scope, summarize

    source_hash = hashlib.sha256()
    records = []
    with DocumentStore(args.database, readonly=True) as store:
        store.connection.execute("BEGIN")
        formal_n = store.connection.execute(
            "SELECT count(*) FROM documents WHERE namespace=? AND json_extract(body,'$.analysis_id')=?",
            (NAMESPACE, ANALYSIS)).fetchone()[0]
        query = """SELECT key,body,updated_ns FROM documents WHERE namespace=?
                   AND json_extract(body,'$.analysis_id')=?
                   AND json_extract(body,'$.origin') IN ('original-data2','original-data3')
                   ORDER BY key"""
        timestamps = []
        for key, body, updated in store.connection.execute(query, (NAMESPACE, ANALYSIS)):
            wire = json.loads(body)
            if not history_in_scope(wire["recsys_type"], wire["post_retain_count"], args.history_scope):
                continue
            source_hash.update(key.encode()+b"\0"+body.encode()+b"\n")
            row = ScenarioStatistics.from_wire(decode_document(wire))
            require(row.mode == "gradation" and row.complete and not row.stale, "Incomplete or unrelated source record")
            require(row.metric_revision == 6 and row.endpoint_complement is True, "Wrong metric convention")
            require(row.active_step is not None and row.active_step > 0, "Figure 5 timing/AUC require positive activity time")
            require(row.pathway_integral is not None and math.isfinite(row.pathway_integral), "Nonfinite I_w")
            require(row.recsys_type in ("StructureM9", "OpinionM9"), "Unsupported recommendation policy")
            records.append(row)
            timestamps.append(updated)
    expected = args.expected_runs if args.expected_runs is not None else (12285 if args.history_scope == "zero" else 24566)
    require(len(records) == expected, f"Source cohort changed: expected {expected}, found {len(records)}")
    require(len({(r.origin, r.name) for r in records}) == len(records), "Duplicate source run identity")
    require(all(r.protocol == records[0].protocol for r in records), "Mixed formal analysis protocol")
    # Positive t_a makes grouped_summary._metrics exactly the plot.py definitions:
    # event_step_mean/max(t_a,1), and clipped piecewise-linear integral of
    # g_index against x_indices/max(t_a,1) on [0,1]. No g_index_mean_active proxy.
    metadata = pd.DataFrame([dict(name=r.name, origin=r.origin, p=r.repost_rate, kh=int(r.post_retain_count),
                                  D_sign="q<alpha" if r.rewiring < r.decay else "q>alpha" if r.rewiring > r.decay else "q=alpha") for r in records])
    tables, contrasts, compositions, cohort_rows, histories = [], [], [], [], []
    summaries = {}
    for cutoff in CUTOFFS:
        data, summary, comparison = summarize(records, cutoff, history_scope=args.history_scope)
        data = data.merge(metadata, on=["name", "origin"], validate="one_to_one")
        summary.insert(0, "cutoff", cutoff)
        comparison.insert(0, "cutoff", cutoff)
        require((summary["n_runs"] == summary["n_observed"]).all(), "Finite metric counts differ from group counts")
        tables.extend(summary.to_dict("records"))
        contrasts.extend(comparison.to_dict("records"))
        compositions.extend(group_composition(data, cutoff))
        histories.extend(history_statistics(data, cutoff))
        for (dimension, categories) in (("policy", ("StructureM9", "OpinionM9")), ("reposting", ("off", "on"))):
            for category in categories:
                for group in ("low", "high"):
                    cohort_rows.append(dict(cutoff=cutoff, dimension=dimension, category=category, group=group,
                                             n=int(((data[dimension] == category) & (data["group"] == group)).sum())))
        summaries[cutoff] = summary

    contrast_frame = pd.DataFrame(contrasts)
    keys = ["comparison", "category", "group", "metric"]
    by_cut = {c: contrast_frame[contrast_frame["cutoff"] == c].fillna({"category": "", "group": ""}).set_index(keys) for c in CUTOFFS}
    changes = []
    for key, first in by_cut[.6].iterrows():
        second = by_cut[.7].loc[key]
        a, b = first["relative_percent"], second["relative_percent"]
        changes.append(dict(zip(keys, key), relative_percent_at_0_6=a, relative_percent_at_0_7=b,
                            change_percentage_points=b-a, direction_reversed=bool(a*b < 0)))
    gaps = []
    for metric in METRICS:
        values = []
        for cutoff in CUTOFFS:
            op = by_cut[cutoff].loc[("high_minus_low", "policy:OpinionM9", "", metric)]["difference"]
            st = by_cut[cutoff].loc[("high_minus_low", "policy:StructureM9", "", metric)]["difference"]
            values.append(float(op-st))
        gaps.append(dict(metric=metric, comparison="Op high-minus-low gap minus St high-minus-low gap",
                         gap_difference_at_0_6=values[0], gap_difference_at_0_7=values[1],
                         direction_reversed=bool(values[0]*values[1] < 0)))
    from experiments.paper.pathway_index.bar_plot import figure5_axis_limits
    limits = figure5_axis_limits(pd.DataFrame(tables))

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("grouped_metric_summary", tables), ("grouped_metric_contrasts", contrasts),
                       ("threshold_changes", changes), ("group_counts", cohort_rows),
                       ("group_composition", compositions), ("history_groups", histories)):
        write_csv(args.output / f"{name}.csv", rows)
    write_csv(args.output / "group_gap_comparison.csv", gaps)
    for cutoff in CUTOFFS:
        plot(summaries[cutoff], cutoff, limits, args.output)
        history_description = "StructureM9 kh=0; OpinionM9 kh=0" if args.history_scope == "zero" else "StructureM9 kh=0; OpinionM9 pooled kh=0,2,6"
        caption = (f"Formal analysis {ANALYSIS}; data2+data3; history scope={args.history_scope}; {history_description}; n={len(records)}. "
                   f"Descriptive low/high I_w split at {cutoff:g}, with endpoint completion retained. "
                   "Top row: StructureM9 and OpinionM9; bottom row: p=0 (!R) and pooled p=.1,.25,.5 (R). "
                   "Bars show equally weighted run means and population standard deviations, not confidence intervals. "
                   "Event time is each run's event_step_mean/max(active_step,1). "
                   "AUC integrates the stored I_s curve over normalized time [0,1], using the existing clipped piecewise-linear trapezoid function. "
                   "Both thresholds share the same axes, which include every mean±SD endpoint; no fixed historical y-limit is imposed. "
                   "Empty or nonfinite data are never replaced by zero; current bars have no missing observations. "
                   "See group_counts.csv and group_composition.csv for the unequal group sizes and pooled parameter composition.\n")
        (args.output / f"figure5_iw_{cutoff:.1f}_caption.txt").write_text(caption, encoding="utf-8")

    source_files = ("experiments/paper/pathway_index/plot.py", "experiments/paper/pathway_index/bar_plot.py",
                    "experiments/paper/pathway_index/grouped_summary.py",
                    "experiments/paper/pathway_index/data_utils.py", "src/common/plotting/core.py",
                    "src/common/storage/documents.py", "experiments/paper/analysis/records.py")
    provenance = dict(database=str(args.database.resolve()), analysis_id=ANALYSIS, origins=["original-data2", "original-data3"],
                      formal_all_history_runs=formal_n, selected_runs=len(records), history_scope=args.history_scope,
                      policy_histories={"StructureM9": [0], "OpinionM9": [0] if args.history_scope == "zero" else [0, 2, 6]},
                      selected_document_snapshot_sha256=source_hash.hexdigest(), snapshot_max_updated_ns=max(timestamps),
                      protocol=records[0].protocol.__wire__(),
                      endpoint_complement=True, metric_revision=6, cutoffs=list(CUTOFFS),
                      minimum_active_step=min(r.active_step for r in records),
                      source_counts={f"{origin}:{policy}": n for (origin, policy), n in sorted(Counter((r.origin, r.recsys_type) for r in records).items())},
                      summary="run-weighted arithmetic means and population SD (ddof=0); post-outcome descriptive groups; not paired causal effects",
                      relative_percent="100*(target mean-reference mean)/reference mean",
                      axis_limits=limits, python=platform.python_version(), numpy=np.__version__,
                      matplotlib=matplotlib.__version__, pandas=pd.__version__, script_sha256=sha(__file__),
                      source_sha256={p: sha(MODEL_ROOT/p) for p in source_files})
    write_csv(args.output / "provenance.csv", [dict(property=k, value=json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in provenance.items()])
    write_report(args.output, args.history_scope, args.database, records, cohort_rows, by_cut,
                 changes, gaps, compositions, histories)
    print(json.dumps(dict(selected=len(records), history_scope=args.history_scope, cutoffs=list(CUTOFFS), reversed_comparisons=[r for r in changes if r["direction_reversed"]], axis_limits=limits, output=str(args.output.resolve())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
