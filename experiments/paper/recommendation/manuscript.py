"""Prepare and compose portable figures for staged recommendation interventions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from common.io.json import atomic_json, sha
from common.plotting import setup_paper_params
from ehk.domain import rows_frame, to_wire
from .figure_models import (
    BridgePanel, CaseCurve, CaseLineStyle, CasePanel, FigureLayout, IllustrativeCase,
    PairedEffectRow, PanelSpecifications, PlotLayout, PlotStyle, PolicyPanel, RenderStyle,
    Thresholds, parse_layout, parse_specifications,
)
from experiments.paper.recommendation_shared import analysis as shared_analysis
from experiments.paper.recommendation_shared import bundle as shared_bundle
from experiments.paper.recommendation_shared.storage import read_table
from experiments.paper.recommendation_shared import plot as shared_plot

from .analysis import coverage_by_setting, read_study
from .design import CROSS_SETTINGS

SCHEMA = 1
CASES = (
    IllustrativeCase("rescue_middle", "s_staged_al10_q01_e45_kh0_p10_sim01", "middle", "α=.10, q=.01, p=.10\nr=1; middle"),
    IllustrativeCase("rescue_late", "s_staged_al10_q01_e45_kh0_p10_sim01", "late", "Same task\nr=1; late"),
    IllustrativeCase("reversal_middle", "s_staged_al10_q03_e45_kh0_p10_sim02", "middle", "α=.10, q=.03, p=.10\nr=2; middle"),
)
COLORS = {"hold": "#666666", "switch": "#2171b5", "bridge": "#238b45", "hide_reposts": "#d95f0e"}
CASE_ARMS = ("hold", "switch", "bridge")
CASE_STYLES = {
    "hold": CaseLineStyle((0, (3, 3)), "s", (0, 6), 1.2, 5),
    "switch": CaseLineStyle("-", "o", (2, 6), 1.6, 3),
    "bridge": CaseLineStyle((0, (5, 2, 1, 2)), "^", (4, 6), 1.3, 4),
}
INSERTED_COLOR = "#2171b5"
ZERO_INSERTION_COLOR = "#777777"


def _effect_rows(frame) -> tuple[PairedEffectRow, ...]:
    return tuple(PairedEffectRow(
        task_id=row.task_id, stage=row.stage, setting=row.setting, sensitivity=row.sensitivity,
        influence=row.influence, rewiring=row.rewiring, repost=row.repost, replicate=row.replicate,
        epsilon=row.epsilon, history=row.history, bridge_shown=row.bridge_shown,
        cross_confident=row.cross_confident, updates=row.updates,
        delta_separation=row.delta_separation, delta_excess=row.delta_excess,
    ) for row in frame.itertuples(index=False))


def panel_specifications(bundle) -> PanelSpecifications:
    manifest = shared_bundle.read_bundle_manifest(bundle)
    plan, records = read_study(bundle / "data/study", "staged")
    thresholds = Thresholds(separation=plan.protocol.separation_threshold,
                            excess_concordance=plan.protocol.excess_threshold)
    runs = read_table(bundle / "data/tables", "runs.csv.gz")
    core = runs[runs["sensitivity"] == "core"]
    native = shared_analysis.paired_effects(core, "switch", "hold")
    bridge = shared_analysis.paired_effects(core, "bridge", "switch")
    settings = tuple(sorted({
        (task.metadata.HKParams.Influence, task.metadata.HKParams.RewiringRate)
        for task in plan.tasks if task.sensitivity == "core"
    }))
    panels = {}
    for repost in sorted(core["repost"].unique()):
        panels[f"policy_p{round(100 * repost):02}"] = PolicyPanel(
            title=f"p={repost:g}", settings=settings,
            rows=_effect_rows(native[native["repost"] == repost]),
        )
    for index, (influence, rewiring) in enumerate(CROSS_SETTINGS):
        selected = bridge[(bridge["influence"] == influence) & (bridge["rewiring"] == rewiring)]
        for metric in ("delta_separation", "delta_excess", "dose"):
            panels[f"bridge_{index}_{metric}"] = BridgePanel(
                title=f"α={influence:g}, q={rewiring:g}", metric=metric, rows=_effect_rows(selected),
            )
    for case in CASES:
        if case.task_id not in records or case.stage not in records[case.task_id].stages:
            raise ValueError(f"illustrative case unavailable: {case.task_id}/{case.stage}")
        block = records[case.task_id].stages[case.stage]
        for metric in ("separation", "excess_concordance"):
            panels[f"case_{case.name}_{metric}"] = CasePanel(
                title=case.title, metric=metric, task_id=case.task_id,
                threshold=getattr(thresholds, metric), checkpoint=block.checkpoint.step,
                start_excess=block.checkpoint.pathway.excess_concordance,
                arms={arm: CaseCurve(step=tuple(point.step for point in data.endpoints),
                                     value=tuple(getattr(point.pathway, metric) for point in data.endpoints))
                      for arm, data in block.arms.items()},
            )
    return PanelSpecifications(
        schema=SCHEMA, data_manifest_sha256=sha(bundle / "data/manifest.json"),
        completed=manifest.completed, planned=manifest.planned,
        core_completed=len(set(core["task_id"])), core_planned=sum(t.sensitivity == "core" for t in plan.tasks),
        source_dataset=manifest.source_dataset, plan_fingerprint=plan.plan_fingerprint,
        coverage=tuple(coverage_by_setting(plan, records)), panels=panels, thresholds=thresholds,
        comparison_unit="paired arms within one task and checkpoint; parameter cells have independent initial states",
    )


def default_layout(specs: PanelSpecifications) -> PlotLayout:
    return PlotLayout(
        schema=SCHEMA, style=PlotStyle(colors=dict(COLORS)), figures={
            "staged_policy_effects": FigureLayout(
                width=7.1, height=4.4, rows=1, columns=3, left=.14, right=.87, bottom=.08, top=.87,
                hspace=.3, wspace=.08, panels=tuple(name for name in specs.panels if name.startswith("policy_"))),
            "staged_bridge_effects": FigureLayout(
                width=7.1, height=5.9, rows=3, columns=4, left=.085, right=.985, bottom=.16, top=.87,
                hspace=.6, wspace=.35, panels=tuple(f"bridge_{index}_{metric}" for metric in
                                                 ("delta_separation", "delta_excess", "dose") for index in range(4))),
            "staged_intervention_cases": FigureLayout(
                width=7.1, height=4.9, rows=2, columns=3, left=.085, right=.985, bottom=.18, top=.82,
                hspace=.43, wspace=.22, panels=tuple(f"case_{case.name}_{metric}" for metric in
                                                   ("separation", "excess_concordance") for case in CASES)),
        },
    )


def _decorate(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)


def _bridge_limits(specs):
    limits = {}
    for metric in ("delta_separation", "delta_excess"):
        values = [getattr(row, metric) for panel in specs.panels.values()
                  if panel.kind == "bridge" and panel.metric == metric for row in panel.rows]
        low, high = min([0, *values]), max([0, *values])
        padding = max(0.02, (high - low) * 0.15)
        limits[metric] = low - padding, high + padding
    rates = [1000 * row.bridge_shown / row.updates
             for panel in specs.panels.values() if panel.kind == "bridge" and panel.metric == "dose"
             for row in panel.rows]
    largest = max([0, *rates])
    limits["dose"] = -max(1, largest * 0.035), max(5, largest * 1.2)
    return limits


def _case_limits(specs):
    values = [value for panel in specs.panels.values()
              if panel.kind == "case" and panel.metric == "excess_concordance"
              for data in panel.arms.values() for value in data.value]
    thresholds = [panel.threshold for panel in specs.panels.values()
                  if panel.kind == "case" and panel.metric == "excess_concordance"]
    low, high = min([0, *values, *thresholds]), max([0, *values, *thresholds])
    padding = max(0.01, (high - low) * 0.08)
    return {"separation": (-0.025, 1.025), "excess_concordance": (low - padding, high + padding)}


def _figure_style(specs: PanelSpecifications, layout: PlotLayout) -> RenderStyle:
    panel_limits = {}
    for name, panel in specs.panels.items():
        if (isinstance(panel, BridgePanel) and panel.metric != "dose"
                and any(name.startswith(f"bridge_{column}_") for column in layout.style.bridge_detail_columns)):
            values = [getattr(row, panel.metric) for row in panel.rows]
            low, high = min([0, *values]), max([0, *values])
            padding = max(.005, (high - low) * .15)
            panel_limits[panel.title, panel.metric] = low - padding, high + padding
    base = layout.style
    return RenderStyle(
        font_size=base.font_size, marker_size=base.marker_size, colors=base.colors,
        heatmap_limit=base.heatmap_limit, bridge_detail_columns=base.bridge_detail_columns,
        bridge_limits=_bridge_limits(specs), case_limits=_case_limits(specs), bridge_panel_limits=panel_limits,
    )


def _swarm_positions(ax, center, values, marker_size, halfwidth=0.33):
    """Pack points horizontally in display space without changing any y value."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.array([]), marker_size
    ys = ax.transData.transform(np.column_stack((np.zeros(len(values)), values)))[:, 1]
    origin = ax.transData.transform((center, 0))[0]
    width = ax.transData.transform((center + halfwidth, 0))[0] - origin
    size = marker_size
    while True:
        # Include the marker outline and a small visible gap in the collision radius.
        diameter = (np.sqrt(size) + 0.55 + 0.35) * ax.figure.dpi / 72
        placed = []
        offsets = np.zeros(len(values))
        for index in np.argsort(ys, kind="stable"):
            neighbours = [(x, y) for x, y in placed if abs(ys[index] - y) < diameter]
            candidates = [0.0]
            for x, y in neighbours:
                clearance = np.sqrt(max(0, diameter ** 2 - (ys[index] - y) ** 2)) + 1e-6
                candidates.extend((x - clearance, x + clearance))
            candidates.sort(key=lambda x: (abs(x), x))
            valid = [x for x in candidates if abs(x) <= width and all(
                (x - previous_x) ** 2 + (ys[index] - previous_y) ** 2 >= diameter ** 2
                for previous_x, previous_y in neighbours)]
            if not valid:
                break
            offsets[index] = valid[0]
            placed.append((valid[0], ys[index]))
        else:
            pixels = np.column_stack((origin + offsets, ys))
            return ax.transData.inverted().transform(pixels)[:, 0], size
        size *= 0.85
        if size < 0.05:
            raise ValueError("bridge swarm is too dense; increase the figure width")


def _case_overlap_note(spec):
    """Describe equality at recorded observations, without estimating a split time."""
    arms = spec.arms
    if not all(arm in arms for arm in CASE_ARMS):
        return None
    hold, switch, bridge = (arms[arm] for arm in CASE_ARMS)
    if hold != switch or bridge.step != hold.step:
        return None
    equal = np.asarray(bridge.value) == np.asarray(hold.value)
    first_different = np.flatnonzero(~equal)
    if not len(first_different):
        return "Hold = switch = bridge\nat all recorded steps"
    if first_different[0] == 0:
        return "Hold = switch\nat all recorded steps"
    last_shared = hold.step[first_different[0] - 1]
    return f"Hold = switch throughout\nAll coincide through recorded\nstep t = {last_shared}"


def _draw_panel(fig, ax, spec, style, *, compact=False):
    kind = spec.kind
    if kind == "policy":
        matrix = np.full((len(spec.settings), 3), np.nan)
        counts = np.zeros_like(matrix, dtype=int)
        rows = rows_frame(spec.rows, PairedEffectRow)
        for i, (influence, rewiring) in enumerate(spec.settings):
            cell = rows[(rows["influence"] == influence) & (rows["rewiring"] == rewiring)]
            for j, phase in enumerate(("early", "middle", "late")):
                values = cell.loc[cell["stage"] == phase, "delta_separation"].dropna()
                if not values.empty:
                    matrix[i, j], counts[i, j] = values.mean(), len(values)
        limit = style.heatmap_limit
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            text = f"{value:+.3f}\nn={counts[i,j]}" if np.isfinite(value) else "unavailable"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 * limit else "black")
        ax.set(xticks=range(3), xticklabels=("Early", "Middle", "Late"), yticks=range(len(matrix)),
               yticklabels=[f"({a:g}, {q:g})" for a, q in spec.settings], ylabel="(α, q)")
        if not compact:
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, label="ΔP: switch − hold")
    elif kind == "bridge":
        rows = rows_frame(spec.rows, PairedEffectRow)
        metric = spec.metric
        ax.set(xlim=(-0.45, 2.48), xticks=range(3), xticklabels=("0", ".10", ".25"), xlabel="Repost probability p",
               ylabel={"delta_separation": "ΔP: bridge − switch", "delta_excess": "ΔE: bridge − switch",
                       "dose": "Bridge slots\n/ 1,000 updates"}[metric])
        ax.set_ylim(style.bridge_panel_limits.get((spec.title, metric), style.bridge_limits[metric]))
        for j, repost in enumerate((0.0, 0.1, 0.25)):
            cell = rows[rows["repost"] == repost].sort_values("replicate") if not rows.empty else rows
            if cell.empty:
                continue
            values = (1000 * cell["bridge_shown"] / cell["updates"]) if metric == "dose" else cell[metric]
            active = cell["bridge_shown"] > 0
            # Only identical zero-insertion observations are collapsed. Their
            # multiplicity is explicit; every inserted observation remains a dot.
            points = [(value, True, 1) for value in values[active]]
            for value, count in values[~active].value_counts(sort=False).items():
                points.append((value, False, count))
            positions, size = _swarm_positions(ax, j - 0.08, [p[0] for p in points], style.marker_size)
            for present, color in ((True, INSERTED_COLOR), (False, ZERO_INSERTION_COLOR)):
                selected = np.array([point[1] == present for point in points])
                if selected.any():
                    ax.scatter(positions[selected], np.array([p[0] for p in points])[selected],
                               s=size if present else max(size, 12), facecolor=color if present else "white",
                               edgecolor=color, linewidth=0.55 if present else 0.8, zorder=3)
            for x, (value, present, count) in zip(positions, points):
                if count > 1:
                    below = metric == "delta_separation"
                    ax.annotate(f"×{count}", (x, value), xytext=(0, -6 if below else 6), textcoords="offset points",
                                ha="center", va="top" if below else "bottom", fontsize=6, color=ZERO_INSERTION_COLOR)
            # The summary occupies its own right-hand lane and cannot fill a
            # zero-insertion circle, even when the mean is exactly zero.
            ax.plot([j + 0.31, j + 0.43], [values.mean()] * 2, color="black", lw=1.4, zorder=4)
            if metric == "delta_separation":
                ax.text(j, 0.96, f"{int(active.sum())}/{len(cell)}", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7)
        ax.axhline(0, color="0.6", lw=0.7)
        _decorate(ax)
    else:
        metric = spec.metric
        for arm in CASE_ARMS:
            if arm not in spec.arms:
                continue
            data = spec.arms[arm]
            ax.plot(data.step, data.value, label=shared_plot.label(arm), color=style.colors[arm],
                    ls=CASE_STYLES[arm].ls, marker=CASE_STYLES[arm].marker, markevery=CASE_STYLES[arm].markevery,
                    lw=CASE_STYLES[arm].lw, zorder=CASE_STYLES[arm].zorder, markersize=2.8, markeredgewidth=0.65,
                    markerfacecolor="white" if arm == "hold" else style.colors[arm])
        ax.axhline(spec.threshold, color="0.6", ls=":", lw=0.8)
        ax.set(xlabel="Simulation step", ylabel="Opinion separation P" if metric == "separation" else "Excess concordance E")
        ax.set_ylim(style.case_limits[metric])
        note = _case_overlap_note(spec)
        if note:
            ax.text(0.035, 0.06 if metric == "separation" else 0.94, note, transform=ax.transAxes,
                    fontsize=6, va="bottom" if metric == "separation" else "top", linespacing=1.3,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
        if not compact:
            ax.legend(fontsize=7)
        _decorate(ax)
    ax.set_title(spec.title, fontsize=8)


def prepare(bundle, output):
    """Save self-contained plot data and an editable layout, without panel images."""
    bundle, output = Path(bundle), Path(output)
    specs = panel_specifications(bundle)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plot_data.json", to_wire(specs))
    layout_path = output / "layout.json"
    if not layout_path.exists():
        atomic_json(layout_path, to_wire(default_layout(specs)))
    return specs


def render_panels(bundle, output):
    """Optionally export independent SVG/PDF/PNG alongside the compact plot data."""
    bundle, output = Path(bundle), Path(output)
    specs = prepare(bundle, output)
    layout_path = output / "layout.json"
    style = _figure_style(specs, parse_layout(json.loads(layout_path.read_text(encoding="utf-8"))))
    setup_paper_params()
    plt.rcParams.update({"font.size": style.font_size, "svg.fonttype": "none"})
    for name, spec in specs.panels.items():
        fig, ax = plt.subplots(figsize=(4.0, 4.8 if spec.kind == "policy" else 2.8), layout="constrained")
        _draw_panel(fig, ax, spec, style)
        shared_plot._save_panel(fig, output / "panels" / name)
    atomic_json(output / "panels/manifest.json", {
        "schema": SCHEMA, "data_manifest_sha256": specs.data_manifest_sha256,
        "plot_data": "../plot_data.json", "specifications_sha256": sha(output / "plot_data.json"), "renderer_sha256": sha(__file__),
        "portable_bundle": str(bundle.resolve()),
        "files": {p.name: sha(p) for p in sorted((output / "panels").iterdir()) if p.suffix in (".svg", ".pdf", ".png")},
    })
    return specs


def compose(directory):
    """Compose vector figures from saved panel data; no dataset or bundle access."""
    directory = Path(directory)
    data_path = directory / "plot_data.json"
    specs = parse_specifications(json.loads(data_path.read_text(encoding="utf-8")))
    layout = parse_layout(json.loads((directory / "layout.json").read_text(encoding="utf-8")))
    if specs.schema != SCHEMA or layout.schema != SCHEMA:
        raise ValueError("unsupported panel or layout schema")
    style = _figure_style(specs, layout)
    setup_paper_params()
    plt.rcParams.update({"font.size": style.font_size, "axes.labelsize": style.font_size,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    prefix = "Preliminary: " if specs.completed < specs.planned else ""
    titles = {
        "staged_policy_effects": "Mean paired switching effects",
        "staged_bridge_effects": "Bridge effects and actual insertion",
        "staged_intervention_cases": "Paired intervention responses",
    }
    for name, config in layout.figures.items():
        fig, axes = plt.subplots(config.rows, config.columns, figsize=(config.width, config.height), squeeze=False)
        fig.subplots_adjust(left=config.left, right=config.right, bottom=config.bottom, top=config.top,
                            hspace=config.hspace, wspace=config.wspace)
        for index, ax in enumerate(axes.flat):
            if index >= len(config.panels):
                ax.set_visible(False)
                continue
            panel_name = config.panels[index]
            _draw_panel(fig, ax, specs.panels[panel_name], style, compact=True)
            title = specs.panels[panel_name].title
            ax.set_title("")
            if name == "staged_intervention_cases":
                title = f"{title}; t={specs.panels[panel_name].checkpoint}" if index < 3 else ""
            ax.set_title(f"({chr(97 + index)}) {title}", loc="left", fontsize=8)
            if index % config.columns:
                ax.set_ylabel("")
                if name != "staged_bridge_effects":
                    ax.tick_params(axis="y", labelleft=False)
            if name != "staged_policy_effects" and index < (config.rows - 1) * config.columns:
                ax.set_xlabel("")
        fig.suptitle(prefix + titles[name], fontsize=10, y=0.98)
        if name == "staged_policy_effects":
            fig.colorbar(axes[0][0].images[0], cax=fig.add_axes([0.89, 0.2, 0.018, 0.56]), label="ΔP: switch − hold")
        elif name == "staged_bridge_effects":
            handles = [
                Line2D([], [], marker="o", ls="", color=INSERTED_COLOR, markersize=4, label="≥1 bridge slot inserted"),
                Line2D([], [], marker="o", ls="", color=ZERO_INSERTION_COLOR, markerfacecolor="white",
                       markersize=4, label="0 bridge slots (×n = repeats)"),
                Line2D([], [], color="black", lw=1.4, label="All-task mean (right)"),
            ]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=7)
        else:
            handles = [Line2D([], [], color=style.colors[arm],
                              ls=CASE_STYLES[arm].ls, marker=CASE_STYLES[arm].marker, lw=CASE_STYLES[arm].lw, zorder=CASE_STYLES[arm].zorder,
                              markersize=3, markerfacecolor="white" if arm == "hold" else style.colors[arm],
                              label=shared_plot.label(arm)) for arm in CASE_ARMS]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=7)
        shared_plot._save_panel(fig, directory / "figures" / name)
    atomic_json(directory / "figure_manifest.json", {
        "schema": SCHEMA, "completed": specs.completed, "planned": specs.planned,
        "data_manifest_sha256": specs.data_manifest_sha256,
        "plot_data": data_path.relative_to(directory).as_posix(), "plot_data_sha256": sha(data_path),
        "panel_specifications_sha256": sha(data_path), "layout_sha256": sha(directory / "layout.json"),
        "renderer_sha256": sha(__file__), "source_dataset": specs.source_dataset,
        "plan_fingerprint": specs.plan_fingerprint,
        "files": {p.name: sha(p) for p in sorted((directory / "figures").iterdir()) if p.suffix in (".svg", ".pdf", ".png")},
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Save compact plot data and layout without independent panel files")
    preparation.add_argument("--bundle", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    panels = commands.add_parser("panels", help="Optionally save independent editable panel files")
    panels.add_argument("--bundle", type=Path, required=True)
    panels.add_argument("--output", type=Path, required=True)
    composition = commands.add_parser("compose", help="Compose figures from panel data and layout only")
    composition.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.bundle, args.output)
    elif args.command == "panels":
        render_panels(args.bundle, args.output)
    else:
        compose(args.directory)


if __name__ == "__main__":
    main()
