"""Render recorded storage measurements without running benchmark actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from statistics import median

BACKENDS = ("current", "tinymongo", "indexed_split")
ACTIONS = (
    "load",
    "point_metadata",
    "point_full",
    "selective_metadata",
    "scan_metadata",
    "scan_full",
    "nested_metadata",
    "state_update",
    "replace_full",
)
QUERY_ACTIONS = {"selective_metadata", "scan_metadata", "scan_full", "nested_metadata", "scope_metadata", "scope_full"}
ACTION_NAMES = {
    "load": "载入 put / 文档",
    "point_metadata": "按键仅元数据 / 次",
    "point_full": "按键完整文档 / 次",
    "selective_metadata": "条件筛选元数据 / 查询",
    "scan_metadata": "扫描元数据 / 查询",
    "scan_full": "扫描完整文档 / 查询",
    "nested_metadata": "嵌套条件元数据 / 查询",
    "state_update": "状态更新 / 文档",
    "replace_full": "完整文档覆盖 / 次",
}
PLOT_NAMES = {
    "load": "Load: put/document",
    "point_metadata": "Point: metadata",
    "point_full": "Point: full",
    "selective_metadata": "Selective: metadata",
    "scan_metadata": "Scan: metadata",
    "scan_full": "Scan: full",
    "nested_metadata": "Nested: metadata",
    "state_update": "State update",
    "replace_full": "Full replacement",
}
COLORS = {"current": "#4477AA", "tinymongo": "#EE7733", "indexed_split": "#228833"}
MIB = 1024**2


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _values(values):
    return [number for value in values if (number := _number(value)) is not None]


def _median(values):
    values = _values(values)
    return median(values) if values else None


def _quantile(values, probability):
    values = sorted(_values(values))
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def _format(value, digits=3, scale=1):
    number = _number(value)
    return "—" if number is None else f"{number / scale:,.{digits}f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _kind(info):
    kind = info.get("data_kind", "未提供")
    if "synthetic" in kind:
        return "校准合成数据，数值不是模型模拟结果"
    if "transformed_actual" in kind:
        return "真实已完成摘要的转换，保留原数值"
    return str(kind)


def _read_records(directory):
    records, excluded, warnings = [], [], []
    for path in sorted((directory / "results").glob("*.json")):
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict):
                raise ValueError("worker result is not an object")
        except (OSError, ValueError) as error:
            warnings.append(f"{path.name}: {error}")
            continue
        record = {**record, "_file": path.name}
        if any(not record.get(key) for key in ("backend", "profile", "action")):
            warnings.append(f"{path.name}: 缺少 backend/profile/action，未进入汇总")
            continue
        if record.get("status") != "ok":
            excluded.append(record)
        else:
            records.append(record)
    # A retried worker must not become a second independent process repetition.
    identities = {}
    for record in records + excluded:
        repeat = record.get("repeat", record["_file"])
        key = (record["profile"], record["backend"], record["action"], str(repeat))
        identities.setdefault(key, []).append(record)
    duplicates = {key for key, items in identities.items() if len(items) > 1}
    if duplicates:
        warnings.extend(f"重复 worker 身份 {_cell(key)}，相关文件均未进入数值汇总" for key in sorted(duplicates))
        records = [
            r
            for r in records
            if (r["profile"], r["backend"], r["action"], str(r.get("repeat", r["_file"]))) not in duplicates
        ]
        excluded = [
            r
            for r in excluded
            if (r["profile"], r["backend"], r["action"], str(r.get("repeat", r["_file"]))) not in duplicates
        ]
    return records, excluded, warnings


def _phases(record):
    raw = record.get("operation_seconds")
    if isinstance(raw, list) and raw and len(_values(raw)) == len(raw):
        raw = _values(raw)
        if record["action"] in QUERY_ACTIONS:
            yield "first", raw[:1]
            if len(raw) > 1:
                yield "steady", raw[1:]
        else:
            yield "all", raw
    else:
        # Aggregate p50 values cannot reconstruct the first and subsequent calls.
        yield "unpartitioned", None


def summarize(records, plan):
    """One row per action/phase; process repetitions remain the aggregation unit."""
    grouped = {}
    for record in records:
        for phase, samples in _phases(record):
            key = (record["profile"], record["backend"], record["action"], phase)
            grouped.setdefault(key, []).append((record, samples))
    rows = []
    for (profile, backend, action, phase), entries in sorted(grouped.items()):
        process_p50, process_p95, process_seconds, operation_counts = [], [], [], []
        for record, samples in entries:
            latency = record.get("latency") or {}
            process_p50.append(
                _quantile(samples, 0.5) * 1000 if samples is not None else _number(latency.get("p50_ms"))
            )
            process_p95.append(
                _quantile(samples, 0.95) * 1000 if samples is not None else _number(latency.get("p95_ms"))
            )
            process_seconds.append(sum(samples) if samples is not None else _number(latency.get("seconds")))
            operation_counts.append(len(samples) if samples is not None else _number(latency.get("operations")))
        members = [r for r, _ in entries]
        info = plan.get("profiles", {}).get(profile, {})
        counts = _values(r.get("count") for r in members)
        p50 = _values(process_p50)
        deltas = [
            max(0, peak - base)
            for r in members
            if (peak := _number(r.get("peak_rss_action_bytes"))) is not None
            and (base := _number(r.get("baseline_rss_bytes"))) is not None
        ]
        row = {
            "profile": profile,
            "data_kind": info.get("data_kind"),
            "backend": backend,
            "action": action,
            "phase": phase,
            "successful_processes": len(entries),
            "planned_repetitions": plan.get("repetitions", 3),
            "profile_count": info.get("count"),
            "profile_measured_count": info.get("measured_count"),
            "worker_count_min": min(counts) if counts else None,
            "worker_count_max": max(counts) if counts else None,
            "operation_count": sum(_values(operation_counts)) if _values(operation_counts) else None,
            "process_seconds_median": _median(process_seconds),
            "process_seconds_min": min(_values(process_seconds)) if _values(process_seconds) else None,
            "process_seconds_max": max(_values(process_seconds)) if _values(process_seconds) else None,
            "operation_p50_ms_median": _median(p50),
            "operation_p50_ms_min": min(p50) if p50 else None,
            "operation_p50_ms_max": max(p50) if p50 else None,
            "operation_p95_ms_median": _median(process_p95),
            "end_to_end_seconds_median": _median(r.get("end_to_end_seconds") for r in members),
            "baseline_rss_bytes_median": _median(r.get("baseline_rss_bytes") for r in members),
            "action_peak_rss_bytes_median": _median(r.get("peak_rss_action_bytes") for r in members),
            "action_peak_rss_delta_bytes_median": _median(deltas),
            "process_peak_rss_bytes_median": _median(r.get("peak_rss_bytes") for r in members),
            "parent_monitored_peak_rss_bytes_median": _median(
                r.get("parent_monitored_peak_rss_bytes") for r in members
            ),
            "files_apparent_bytes_median": _median((r.get("files") or {}).get("apparent_bytes") for r in members),
            "files_allocated_bytes_median": _median((r.get("files") or {}).get("allocated_bytes") for r in members),
            "open_seconds_median": _median(r.get("open_seconds") for r in members),
            "checkpoint_seconds_median": _median(r.get("checkpoint_seconds") for r in members),
            "source_files": json.dumps([r["_file"] for r in members]),
        }
        rows.append(row)
    return rows


def _action_order(actions):
    return sorted(actions, key=lambda action: (ACTIONS.index(action) if action in ACTIONS else len(ACTIONS), action))


def _figures(directory, rows, records, profiles, backends):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    profiles = profiles or ["no_measurements"]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(
        1, len(profiles), squeeze=False, figsize=(max(5, 4.9 * len(profiles)), 7.4), constrained_layout=True
    )
    keys = sorted(
        {(r["action"], r["phase"]) for r in rows},
        key=lambda key: (
            _action_order({r["action"] for r in rows}).index(key[0]),
            {"all": 0, "first": 1, "steady": 2, "unpartitioned": 3}.get(key[1], 4),
        ),
    )
    lookup = {(r["profile"], r["backend"], r["action"], r["phase"]): r for r in rows}
    bar_width = 0.75 / max(1, len(backends))
    for column, profile in enumerate(profiles):
        axis = axes[0, column]
        for index, backend in enumerate(backends):
            for location, (action, phase) in enumerate(keys):
                row = lookup.get((profile, backend, action, phase), {})
                value = row.get("operation_p50_ms_median")
                if value is None or value <= 0:
                    continue
                low = row.get("operation_p50_ms_min", value)
                high = row.get("operation_p50_ms_max", value)
                position = location + (index - (len(backends) - 1) / 2) * bar_width
                axis.barh(position, value, height=bar_width * 0.88, color=COLORS.get(backend, "#888888"))
                axis.errorbar(
                    value,
                    position,
                    xerr=[[max(0, value - low)], [max(0, high - value)]],
                    color="#333333",
                    capsize=2,
                    linewidth=0.7,
                )
        labels = [
            PLOT_NAMES.get(a, a)
            + ({"first": " / first", "steady": " / subsequent", "unpartitioned": " / unsplit"}.get(p, ""))
            for a, p in keys
        ]
        axis.set_yticks(np.arange(len(keys)), labels if column == 0 else [""] * len(keys))
        axis.invert_yaxis()
        if any(patch.get_width() > 0 for patch in axis.patches):
            axis.set_xscale("log")
        else:
            axis.text(0.5, 0.5, "No positive latency measurements", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(profile)
        axis.set_xlabel("Median process p50 (ms; log scale)")
        axis.grid(axis="x", alpha=0.2)
    figure.legend(
        handles=[Patch(color=COLORS.get(b, "#888888"), label=b) for b in backends],
        loc="outside lower center",
        ncol=max(1, len(backends)),
        frameon=False,
    )
    figure.suptitle("Operation latency; whiskers show process p50 min/max")
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"performance.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)

    actions = _action_order({r["action"] for r in records})
    figure, axes = plt.subplots(
        2, len(profiles), squeeze=False, figsize=(max(5, 4.9 * len(profiles)), 9), constrained_layout=True
    )
    for column, profile in enumerate(profiles):
        for metric_row, delta in enumerate((False, True)):
            axis = axes[metric_row, column]
            for index, backend in enumerate(backends):
                for location, action in enumerate(actions):
                    selected = [
                        r for r in records if (r["profile"], r["backend"], r["action"]) == (profile, backend, action)
                    ]
                    values = []
                    for record in selected:
                        peak, base = (
                            _number(record.get("peak_rss_action_bytes")),
                            _number(record.get("baseline_rss_bytes")),
                        )
                        if peak is not None and (not delta or base is not None):
                            values.append(max(0, peak - base) / MIB if delta else peak / MIB)
                    if not values:
                        continue
                    value = median(values)
                    position = location + (index - (len(backends) - 1) / 2) * bar_width
                    axis.barh(position, value, height=bar_width * 0.88, color=COLORS.get(backend, "#888888"))
                    axis.errorbar(
                        value,
                        position,
                        xerr=[[value - min(values)], [max(values) - value]],
                        color="#333333",
                        capsize=2,
                        linewidth=0.7,
                    )
            axis.set_yticks(
                np.arange(len(actions)), [PLOT_NAMES.get(a, a) for a in actions] if column == 0 else [""] * len(actions)
            )
            axis.invert_yaxis()
            axis.set_title(profile)
            axis.set_xlabel("Action peak RSS minus baseline (MiB)" if delta else "Sampled action peak RSS (MiB)")
            axis.grid(axis="x", alpha=0.2)
            if not axis.patches:
                axis.text(0.5, 0.5, "No action RSS measurements", ha="center", va="center", transform=axis.transAxes)
    figure.legend(
        handles=[Patch(color=COLORS.get(b, "#888888"), label=b) for b in backends],
        loc="outside lower center",
        ncol=max(1, len(backends)),
        frameon=False,
    )
    figure.suptitle("Whole-action sampled RSS; bars=median, whiskers=min/max across processes")
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"memory.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _recovery_rows(plan, directory):
    rows = []
    for backend, source in (plan.get("recovery") or {}).items():
        try:
            if isinstance(source, dict):
                result = source
            else:
                path = Path(source)
                if not path.is_absolute():
                    path = next((p for p in (directory / path, Path.cwd() / path) if p.is_file()), directory / path)
                result = json.loads(path.read_text())
            rows.append(
                (
                    backend,
                    result.get("passed_rounds"),
                    result.get("planned_rounds"),
                    result.get("all_passed"),
                    str(source),
                )
            )
        except (OSError, TypeError, ValueError) as error:
            rows.append((backend, None, None, "未读取", str(error)))
    return rows


def render(run_dir):
    directory = Path(run_dir).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    records, excluded, warnings = _read_records(directory)
    rows = summarize(records, plan)
    profiles = list(plan.get("profiles", {})) or sorted({r["profile"] for r in records})
    backends = list(plan.get("backends", BACKENDS))
    repetitions = int(plan.get("repetitions", 3))
    actions = list(dict.fromkeys(("load", *plan.get("actions", ACTIONS))))
    expected = set(product(profiles, backends, actions, range(repetitions)))
    done = {
        (r["profile"], r["backend"], r["action"], r.get("repeat"))
        for r in records + excluded
        if r.get("status") in ("ok", "not_applicable")
    }
    missing = expected - done
    smoke = repetitions < 3 or any(
        info.get("measured_count", info.get("count")) != info.get("count") for info in plan.get("profiles", {}).values()
    )
    csv_path = directory / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = list(rows[0]) if rows else ["profile", "backend", "action", "phase", "successful_processes"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _figures(directory, rows, records, profiles, backends)

    lines = [
        "# 隔离存储基准结果",
        "",
        "本报告仅汇总已有 worker JSON，没有执行新的性能测量。数值仅取 status=ok 的结果；不适用和失败项单列。",
        "",
    ]
    if smoke:
        lines.extend(["本目录是小规模或单重复流程验证，不能作为正式全规模、多独立进程重复的性能结论。", ""])
    lines.extend(
        [
            f"计划每项 {repetitions} 个独立进程重复，同进程条件查询计划重复 {plan.get('query_repetitions', '未提供')} 次。"
            f"已读取 {len(records)} 个成功 worker、{len(excluded)} 个非成功 worker；计划中尚缺 {len(missing)} 个完成或不适用结果。",
            "",
            "## 数据与配置",
            "",
            "| profile | 原任务规模 | 本次实测文档数 | 数据性质 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for profile in profiles:
        info = plan.get("profiles", {}).get(profile, {})
        lines.append(
            f"| {_cell(profile)} | {_format(info.get('count'), 0)} | {_format(info.get('measured_count'), 0)} | {_cell(_kind(info))} |"
        )
    lines.extend(
        [
            "",
            "full_paper 和 versioned_2x 的校准合成规模不是恢复得到的真实全量论文统计；实际研究来源及合成假设见 [plan.json](plan.json)。",
            "",
            "三个方案的底层都是 SQLite。这里比较文档接口、索引、编码和数组布局。WAL 与 synchronous=FULL 是本次受控配置，不能当作第三方库未配置时的默认承诺。",
            "",
            "每项在新进程中运行，但操作系统 page cache 未受控，因此 first 不是冷盘测量。first 是进程内第一次同类条件查询，steady 是同一进程的后续查询；默认两次查询时只有一个后续样本。没有原始 operation_seconds 的记录仅列 unpartitioned，不重建 first/steady。",
            "",
            "## 延迟",
            "",
            "表格给出各进程操作 p50 的中位数，单位 ms；括号是成功独立进程数。summary.csv 保留各进程 p50 的最小/最大值和各进程 p95 的中位数。未合并操作样本来伪增独立重复次数。默认条件查询中，每个进程的 first 和 steady 各只有一个观测值；此时 p95/p99 不是尾延迟估计，应比较各阶段独立进程重复的中位数和实测范围。",
            "",
        ]
    )
    lookup = {(r["profile"], r["backend"], r["action"], r["phase"]): r for r in rows}
    for profile in profiles:
        lines.extend(
            [
                f"### {_cell(profile)}",
                "",
                "| 操作与阶段 | " + " | ".join(backends) + " |",
                "| --- | " + " | ".join("---:" for _ in backends) + " |",
            ]
        )
        keys = sorted(
            {(r["action"], r["phase"]) for r in rows if r["profile"] == profile},
            key=lambda k: (actions.index(k[0]) if k[0] in actions else len(actions), k[1]),
        )
        for action, phase in keys:
            cells = []
            for backend in backends:
                row = lookup.get((profile, backend, action, phase))
                cells.append(
                    "—" if row is None else f"{_format(row['operation_p50_ms_median'])} ({row['successful_processes']})"
                )
            lines.append(f"| {_cell(ACTION_NAMES.get(action, action))} · {phase} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "![操作延迟](performance.png)",
            "",
            "图中横线是独立进程 p50 的实测最小至最大范围，不是置信区间。对数坐标只绘制正延迟。",
            "",
            "## 载入和文件体积",
            "",
            "put 总耗时是逐文档 adapter 调用耗时之和，包含编码、压缩和提交，排除输入生成与 hash。载入循环端到端耗时另含流式生成和输入 hash，结束于载入循环，不含之后的抽查与 checkpoint，不与 put 总耗时混用。文件体积按 worker 实际返回值汇总，不从逻辑数组大小推算。",
            "",
            "| profile / backend | put 总秒 | 载入循环端到端秒 | checkpoint 秒 | 文件逻辑 MiB | 文件分配 MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["action"] == "load":
            lines.append(
                f"| {row['profile']} / {row['backend']} | {_format(row['process_seconds_median'])} | {_format(row['end_to_end_seconds_median'])} | {_format(row['checkpoint_seconds_median'])} | {_format(row['files_apparent_bytes_median'], scale=MIB)} | {_format(row['files_allocated_bytes_median'], scale=MIB)} |"
            )
    lines.extend(
        [
            "",
            "## 内存",
            "",
            "action 峰值来自 worker 的 psutil RSS 采样，当前间隔为 10ms；基线单列。整个进程的 ru_maxrss 可能包含初始化或其他阶段，不能替代 action 峰值，也不能把 action 内存分摊给 first/steady。采样窗口覆盖整个 action，含计时外校验；load 还含生成和 checkpoint，state_update 还含 get(full) 内容验证，因此其 RSS 不是纯状态修改的占用。主要内存对照应看 metadata scan 和 full scan。所有值为 MiB，跨进程重复取中位数。",
            "",
            "| profile / backend / action | 基线 | action 峰值 | 超出基线 | 进程 ru_maxrss |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    seen = set()
    for row in rows:
        identity = (row["profile"], row["backend"], row["action"])
        if identity in seen:
            continue
        seen.add(identity)
        lines.append(
            "| "
            + " / ".join(identity)
            + " | "
            + " | ".join(
                _format(row[field], scale=MIB)
                for field in (
                    "baseline_rss_bytes_median",
                    "action_peak_rss_bytes_median",
                    "action_peak_rss_delta_bytes_median",
                    "process_peak_rss_bytes_median",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "![action 内存](memory.png)",
            "",
            "## 不适用、失败与缺失",
            "",
            "| profile / backend / action | status | 文件 | 原因 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in excluded:
        reason = record.get("reason", record.get("error", "未提供"))
        lines.append(
            f"| {_cell(record['profile'])} / {_cell(record['backend'])} / {_cell(record['action'])} | {_cell(record.get('status', '未提供'))} | {_cell(record['_file'])} | {_cell(reason)} |"
        )
    if not excluded:
        lines.append("| — | — | — | 无非成功结果 |")
    if missing:
        lines.extend(["", f"还有 {len(missing)} 个计划 worker 缺少完成结果，当前汇总为部分结果。"])
    if warnings:
        lines.extend(["", "读取问题：", "", *[f"- {_cell(item)}" for item in warnings]])
    recovery = _recovery_rows(plan, directory)
    lines.extend(
        [
            "",
            "## 独立恢复验证",
            "",
            "恢复验证与性能计时分开。它只针对 writer 子进程 SIGKILL，不模拟断电。",
            "",
            "| backend | 通过轮数 | 计划轮数 | all_passed |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for backend, passed, planned, all_passed, source in recovery:
        lines.append(f"| {_cell(backend)} | {_format(passed, 0)} | {_format(planned, 0)} | {_cell(all_passed)} |")
    if not recovery:
        lines.append("| — | — | — | plan 未附恢复结果 |")
    lines.extend(
        [
            "",
            "外部恢复报告路径及源码身份保留在 plan 的 recovery 字段；本渲染器没有重新运行恢复测试。",
            "",
            "[完整汇总 CSV](summary.csv) · [延迟 PDF](performance.pdf) · [内存 PDF](memory.pdf)",
            "",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "run_dir": str(directory),
        "summary_rows": len(rows),
        "successful_workers": len(records),
        "excluded_workers": len(excluded),
        "missing_workers": len(missing),
        "warnings": warnings,
        "smoke_or_limited": smoke,
        "report": str(directory / "report.md"),
        "summary": str(csv_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
