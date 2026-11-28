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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "e499b1ab5ff7ddc6480c5578127996e1945eecfd78af2f9a605f0fb7b93a267e"
NAMESPACE = "ehk-paper-statistics-v3"
MODEL_ROOT = Path(__file__).resolve().parents[3]
CUTOFFS = (.6, .7)
METRICS = ("rewiring_events", "normalized_rewiring_time", "subjective_index_auc", "closed_triads")
AXIS_LABELS = ("Rewiring events", "Normalized event time", r"AUC of $I_s$", "Closed triads")


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
                                     ("D_sign", ("q<alpha", "q=alpha", "q>alpha"))):
                    for level in levels:
                        count = int((block[axis] == level).sum())
                        rows.append(dict(cutoff=cutoff, dimension=dimension, category=category, group=group,
                                         group_n=len(block), axis=axis, level=level, n=count,
                                         fraction=count/len(block) if len(block) else None))
    return rows


def plot(summary, cutoff, limits, destination):
    from common.plotting import PAPER_WIDTH, plt_figure, plt_save_and_close, setup_paper_params

    setup_paper_params()
    fig, axes = plt_figure(n_row=2, n_col=4, total_width=PAPER_WIDTH*(2/3), hw_ratio=1.35,
                           sharey="col", constrained_layout=False)
    for row, (dimension, categories, labels) in enumerate((
        ("policy", ("StructureM9", "OpinionM9"), ("St.", "Op.")),
        ("reposting", ("off", "on"), ("!R", "R")),
    )):
        for col, (metric, ylabel) in enumerate(zip(METRICS, AXIS_LABELS)):
            ax = axes[row, col]
            for offset, (group, color) in enumerate((("low", "tab:blue"), ("high", "tab:red"))):
                records = [summary[(summary["dimension"] == dimension) & (summary["category"] == category) &
                                   (summary["group"] == group) & (summary["metric"] == metric)].iloc[0]
                           for category in categories]
                require(all(record["n_observed"] > 0 for record in records), "Empty Figure 5 bar")
                ax.bar(np.arange(2)+(offset-.5)*.30, [r["mean"] for r in records], width=.30,
                       yerr=[r["population_std"] for r in records], color=color, alpha=.55,
                       edgecolor="black", linewidth=.7, capsize=2,
                       error_kw=dict(elinewidth=1, capthick=1), label=rf"{group} $I_w$")
            ax.set_xticks(np.arange(2), labels)
            ax.set_ylabel(ylabel)
            ax.set_title(f"({chr(97+row*4+col)})", loc="left")
            ax.set_yscale(limits[metric]["scale"])
            ax.set_ylim(*limits[metric]["limits"])
            if limits[metric]["scale"] == "linear" and metric in ("rewiring_events", "closed_triads"):
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.grid(axis="y", alpha=.25, linestyle="--")
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=1.4, w_pad=.7)
    stem = destination / f"figure5_iw_{cutoff:.1f}"
    plt_save_and_close(fig, str(stem))


def write_report(destination, counts, by_cut, gaps):
    claim_specs = (
        ("高组：R 相对 !R 的换边数", "+34%", ("reposting_on_minus_off", "", "high", "rewiring_events")),
        ("高组：Op 相对 St 的归一化换边时刻", "−32%", ("opinion_minus_structure", "", "high", "normalized_rewiring_time")),
        ("高组：R 相对 !R 的归一化换边时刻", "+41%", ("reposting_on_minus_off", "", "high", "normalized_rewiring_time")),
        ("St：高组相对低组的闭合三元组", "+17%", ("high_minus_low", "policy:StructureM9", "", "closed_triads")),
        ("Op：高组相对低组的闭合三元组", "+123%", ("high_minus_low", "policy:OpinionM9", "", "closed_triads")),
        ("!R：高组相对低组的闭合三元组", "+32%", ("high_minus_low", "reposting:off", "", "closed_triads")),
        ("R：高组相对低组的闭合三元组", "+98%", ("high_minus_low", "reposting:on", "", "closed_triads")),
        ("高组：Op 相对 St 的闭合三元组", "+7%", ("opinion_minus_structure", "", "high", "closed_triads")),
    )
    lines = [
        "# Figure 5：0.6 与 0.7 工作分界复核", "",
        "日期：2026-11-28。读取 data2+data3 的正式主库，分析身份为 `"+ANALYSIS+"`；原始模拟和状态指标均未重算。", "",
        "本次只用 `k_h=0`，共 **12,285** 条完整运行（StructureM9 6,144，OpinionM9 6,141）。data1 没有接入；512 条 P/E 诊断与旧 32 条 pilot 也没有混入。保留当前 `L` 指标及默认终点补项，低组为 `I_w<c`、高组为 `I_w>=c`。分组用于描述路径倾向。", "",
        "## 统计口径与可复现性", "",
        "直接调用模型仓库 `pathway_index/grouped_summary.py:summarize`，与 Figure 5 原制图入口逐项核对。所有选中运行的 `active_step>0`，因此以下表达与原 `plot.py` 的 `max(active_step,1)` 完全相同：", "",
        "- 换边数、最终闭合三元组数直接读主库。换边时刻是每条运行在完整 `[0,T]` 内的 `event_step_mean/active_step`；包括活动时刻之后的事件，没有另截断事件窗口。",
        "- AUC 精确调用 `piecewise_linear_integral_trapz(x_indices/active_step, g_index, 0, 1)`：在保存的自适应点间折线插值、补齐积分边界并截取归一化 `[0,1]`。它不是 `mean(g_index)`，也不是 `g_index_mean_active`。",
        "- 每条运行等权；柱是算术均值，误差棒是总体 SD（`ddof=0`）。没有先对每格取均值，没有事件加权，也没有把 SD 当置信区间。各柱四项指标均无缺失。",
        "- 百分比统一为 `100*(target_mean-reference_mean)/reference_mean`，对应分母见下表比较方向。", "",
        "复现命令（使用模型仓库已安装依赖的环境）：", "",
        "```powershell",
        f"Set-Location {MODEL_ROOT.as_posix()}",
        f"{MODEL_ROOT.as_posix()}/.venv/Scripts/python.exe -m experiments.paper.pathway_metrics.figure5 --output {destination.resolve().as_posix()}",
        "```", "",
        "入口位于模型仓库 `experiments/paper/pathway_metrics/figure5.py`，本展品目录不保留代码副本。默认主库路径是 `E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite`；`--database` 可显式指定输入，`--output` 必填，禁止把源码目录当默认输出。入口核对当前 12,285 条样本，样本数变化会拒绝静默替换；同人数下的内容变化可通过快照 SHA 识别。`provenance.csv` 记录选中 SQLite 文档快照 SHA、协议、来源、源码 SHA 与软件版本。所有 CSV 是显式派生产物，不作输入缓存；没有落地逐运行轨迹或表。", "",
        "## 分组大小与迁移", "",
        "| 条件 | 0.6 低组 | 0.6 高组 | 0.7 低组 | 0.7 高组 | 从高组移入低组 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    count_index = {(r["cutoff"], r["dimension"], r["category"], r["group"]): r["n"] for r in counts}
    for dimension, category, label in (("policy", "StructureM9", "St"), ("policy", "OpinionM9", "Op"),
                                       ("reposting", "off", "!R"), ("reposting", "on", "R")):
        ns = [count_index[c, dimension, category, group] for c in CUTOFFS for group in ("low", "high")]
        lines.append(f"| {label} | {ns[0]:,} | {ns[1]:,} | {ns[2]:,} | {ns[3]:,} | {ns[1]-ns[3]:,} |")
    lines += ["", "总计 847 条，即全体的 6.89%，位于 `[0.6,0.7)`，随提高分界从高组进入低组；原有低组不会反向迁移。", "",
              "这些是运行结束后的条件分组，算法／转发组的参数组成并不相同。!R 仅 `p=0`，共 3,071 条；R 合并 `.1,.25,.5` 三个速率，共 9,214 条。以 0.6 高组为例，`q>alpha` 占 St 的 73.58%、Op 的 65.32%；提高至 0.7 后分别为 81.32%、70.67%。算法间均值差和分界变化因此都含分组组成的影响，不能写作相同初态下的配对干预效应。完整组成见 `group_composition.csv`。", "",
              "## 对照旧正文的数字", "",
              "| 比较 | 旧正文 | 当前 0.6 | 当前 0.7 |", "| --- | ---: | ---: | ---: |"]
    for label, old, key in claim_specs:
        values = [float(by_cut[c].loc[key]["relative_percent"]) for c in CUTOFFS]
        lines.append(f"| {label} | {old} | {values[0]:+.2f}% | {values[1]:+.2f}% |")
    lines += ["", "**对旧正文最重要的方向变化是高组的闭合三元组：Op 相对 St 从旧称 +7% 改为 −14.16%／−16.86%。** 0.6 下均值为 St 18,646、Op 16,006；0.7 下为 St 20,108、Op 16,718。因而当前公平比较支持高组内 St 的三元组均值更大。两分界都如此，不能把这一反向归结为把 0.6 改成 0.7；本次也没有拆解指标修订、历史条件混合及数据覆盖变化各自的贡献。", "",
              "转发组在高组的换边数更多、归一化换边更晚，两分界方向均保持，但幅度不同。Op 高组换边时刻更早也保持，幅度由 −20.19% 收窄至 −15.66%。高组 Op 相对 St 的换边数由 +7.79% 收窄至 +1.89%；AUC 优势由 +2.34% 收窄至 +0.55%。应保留方向与量级的区别，不能称全部百分比对分界稳定。", "",
              "## 0.6 改为 0.7 时，哪些方向发生变化", "",
              "32 项标准组均值对比中，只有一项百分比符号变化：**低组内 Op 相对 St 的归一化换边时刻，−0.73% 变为 +0.38%**。0.6 下为 Op `.260701`、St `.262612`；0.7 下为 Op `.260989`、St `.259993`。这是接近零的数值符号翻转，不作统计效应或机制反转解读，也没有据 SD 推断显著性。其余 31 项标准对比方向不变，详见 `threshold_changes.csv`。", "",
              "旧文称低组“相对稳定”，不能只凭柱高相近直接沿用：低组 Op/St 换边数差为 +6.45%／+11.27%，R/!R 为 +0.47%／+3.04%；换边时刻差虽均在 1% 以内，也没有做等效性或稳定性检验。若扩展到其他指标，低组三元组 Op/St 差为 −41.33%／−39.30%，显然不是所有性质都相近。应明确所说的指标、分组、分界与描述尺度。", "",
              "另外，若比较各条件内部“高组均值−低组均值”的差距大小，算法排序有两项变化（`group_gap_comparison.csv`）：", "",
              "| 高低组差距的算法比较 | 0.6：Op差距−St差距 | 0.7：Op差距−St差距 |", "| --- | ---: | ---: |"]
    for row in gaps:
        if row["direction_reversed"]:
            lines.append(f"| {row['metric']} | {row['gap_difference_at_0_6']:+.6f} | {row['gap_difference_at_0_7']:+.6f} |")
    lines += ["", "这是差距大小的描述性排序变化，不是新增因果估计。AUC 在四个算法／转发条件中仍始终是高组大于低组：0.6 相对增幅约 15.04%–19.23%，0.7 为 20.36%–24.27%。较大的 AUC 只表明在该运行归一化活动时钟内局部模板相似度更高／更前置；不等价于总体极化更大或物理时间更早。", "",
              "## 图形与范围", "",
              "两图保持 Figure 5 的八面板顺序。字体、数学字形和线宽沿用 `common.plotting`；PDF 是矢量导出，PNG 为 300 dpi。两分界和上下行使用相同坐标范围，覆盖全部均值±SD端点，取消原入口的固定 `ylim`。换边数和三元组用对数轴；归一化时刻和 AUC 用线性轴。部分时刻均值减 SD 小于零，完整展示的是对称离散度范围，不是负的事件时刻观测。样本数和统计定义放在外置 caption 中。", "",
              "旧正文 `[ehk/results.tex]` 的 +34%/−32%/+41% 与三元组百分比需要整体同步；换边时序解释末尾引用 `(g)` 也不符，时序面板是 `(f)`，而 `(g)` 是 AUC。本图5复核仅在 artifacts 生成展品，未替换正文图5及其展示文字；SI新增完整诊断另见[路径指标诊断](../ehk_iw_threshold_20261128/README.md)。", "",
              "- `figure5_iw_0.6.pdf/png`、`figure5_iw_0.7.pdf/png`：两版图。",
              "- `grouped_metric_summary.csv`：64 个柱均值／SD及逐指标有效 n。",
              "- `grouped_metric_contrasts.csv`：两分界全部 64 项标准比较。",
              "- `threshold_changes.csv`、`group_gap_comparison.csv`：分界变更与差距排序。",
              "- `group_counts.csv`、`group_composition.csv`：不均衡组数及来源／p／D 组成。",
              "- `provenance.csv`：输入与实现凭证。", ""]
    (destination / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("E:/ResearchData/ehk/original/20261127/primary/stats-v3.sqlite"))
    parser.add_argument("--output", type=Path, required=True, help="Explicit directory for Figure 5 exhibits; no source-tree default")
    parser.add_argument("--expected-runs", type=int, default=12285)
    args = parser.parse_args()
    from common.storage.documents import DocumentStore, decode_document
    from experiments.paper.analysis.records import ScenarioStatistics
    from experiments.paper.pathway_index.grouped_summary import summarize

    source_hash = hashlib.sha256()
    records = []
    with DocumentStore(args.database, readonly=True) as store:
        store.connection.execute("BEGIN")
        formal_n = store.connection.execute(
            "SELECT count(*) FROM documents WHERE namespace=? AND json_extract(body,'$.analysis_id')=?",
            (NAMESPACE, ANALYSIS)).fetchone()[0]
        query = """SELECT key,body,updated_ns FROM documents WHERE namespace=?
                   AND json_extract(body,'$.analysis_id')=?
                   AND json_extract(body,'$.post_retain_count')=0
                   AND json_extract(body,'$.origin') IN ('original-data2','original-data3')
                   ORDER BY key"""
        timestamps = []
        for key, body, updated in store.connection.execute(query, (NAMESPACE, ANALYSIS)):
            source_hash.update(key.encode()+b"\0"+body.encode()+b"\n")
            row = ScenarioStatistics.from_wire(decode_document(json.loads(body)))
            require(row.mode == "gradation" and row.complete and not row.stale, "Incomplete or unrelated source record")
            require(row.metric_revision == 6 and row.endpoint_complement is True, "Wrong metric convention")
            require(row.active_step is not None and row.active_step > 0, "Figure 5 timing/AUC require positive activity time")
            require(row.pathway_integral is not None and math.isfinite(row.pathway_integral), "Nonfinite I_w")
            require(row.recsys_type in ("StructureM9", "OpinionM9"), "Unmatched recommendation policy")
            records.append(row)
            timestamps.append(updated)
    require(len(records) == args.expected_runs, f"Source cohort changed: expected {args.expected_runs}, found {len(records)}")
    require(len({(r.origin, r.name) for r in records}) == len(records), "Duplicate source run identity")
    require(all(r.protocol == records[0].protocol for r in records), "Mixed formal analysis protocol")
    # Positive t_a makes grouped_summary._metrics exactly the plot.py definitions:
    # event_step_mean/max(t_a,1), and clipped piecewise-linear integral of
    # g_index against x_indices/max(t_a,1) on [0,1]. No g_index_mean_active proxy.
    metadata = pd.DataFrame([dict(name=r.name, origin=r.origin, p=r.repost_rate,
                                  D_sign="q<alpha" if r.rewiring < r.decay else "q>alpha" if r.rewiring > r.decay else "q=alpha") for r in records])
    tables, contrasts, compositions, cohort_rows = [], [], [], []
    summaries = {}
    for cutoff in CUTOFFS:
        data, summary, comparison = summarize(records, cutoff)
        data = data.merge(metadata, on=["name", "origin"], validate="one_to_one")
        summary.insert(0, "cutoff", cutoff)
        comparison.insert(0, "cutoff", cutoff)
        require((summary["n_runs"] == summary["n_observed"]).all(), "Finite metric counts differ from group counts")
        tables.extend(summary.to_dict("records"))
        contrasts.extend(comparison.to_dict("records"))
        compositions.extend(group_composition(data, cutoff))
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
    limits = {}
    for metric in METRICS:
        selected = [r for r in tables if r["metric"] == metric]
        lower = min(r["mean"]-r["population_std"] for r in selected)
        upper = max(r["mean"]+r["population_std"] for r in selected)
        log = metric in ("rewiring_events", "closed_triads") and lower > 0
        if log:
            limits[metric] = dict(scale="log", limits=[lower/1.15, upper*1.15])
        else:
            span = upper-min(0., lower)
            limits[metric] = dict(scale="linear", limits=[min(0., lower)-.04*span, upper+.10*span])

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("grouped_metric_summary", tables), ("grouped_metric_contrasts", contrasts),
                       ("threshold_changes", changes), ("group_counts", cohort_rows), ("group_composition", compositions)):
        write_csv(args.output / f"{name}.csv", rows)
    write_csv(args.output / "group_gap_comparison.csv", gaps)
    for cutoff in CUTOFFS:
        plot(summaries[cutoff], cutoff, limits, args.output)
        caption = (f"Formal analysis {ANALYSIS}; data2+data3; kh=0; n={len(records)}. "
                   f"Descriptive low/high I_w split at {cutoff:g}, with endpoint completion retained. "
                   "Top row: StructureM9 and OpinionM9; bottom row: p=0 (!R) and pooled p=.1,.25,.5 (R). "
                   "Bars show equally weighted run means and population standard deviations, not confidence intervals. "
                   "Event time is each run's event_step_mean/max(active_step,1). "
                   "AUC integrates the stored I_s curve over normalized time [0,1], using the existing clipped piecewise-linear trapezoid function. "
                   "Both thresholds share the same axes, which include every mean±SD endpoint; no fixed historical y-limit is imposed. "
                   "Empty or nonfinite data are never replaced by zero; current bars have no missing observations. "
                   "See group_counts.csv and group_composition.csv for the unequal group sizes and pooled parameter composition.\n")
        (args.output / f"figure5_iw_{cutoff:.1f}_caption.txt").write_text(caption, encoding="utf-8")

    source_files = ("experiments/paper/pathway_index/plot.py", "experiments/paper/pathway_index/grouped_summary.py",
                    "experiments/paper/pathway_index/data_utils.py", "src/common/plotting/core.py",
                    "src/common/storage/documents.py", "experiments/paper/analysis/records.py")
    provenance = dict(database=str(args.database.resolve()), analysis_id=ANALYSIS, origins=["original-data2", "original-data3"],
                      formal_all_history_runs=formal_n, selected_kh0_runs=len(records),
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
    write_report(args.output, cohort_rows, by_cut, gaps)
    print(json.dumps(dict(selected=len(records), cutoffs=list(CUTOFFS), reversed_comparisons=[r for r in changes if r["direction_reversed"]], axis_limits=limits, output=str(args.output.resolve())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
