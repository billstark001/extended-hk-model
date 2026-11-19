"""Export observations, prepare plot data, and compose portable manuscript figures.

Only export reads the dataset. Preparation authenticates the local bundle;
composition reads its self-contained plot data and an editable layout.
"""

import argparse
import gzip
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from common.io.json import atomic_json, sha
from common.plotting import setup_paper_params
from common.plotting.figures import save

from .analysis import coverage_by_setting, read_study
from .design import CROSS_SETTINGS
from .plot import analyze, label

SCHEMA = 1
CASES = (
    ("rescue_middle", "s_staged_al10_q01_e45_kh0_p10_sim01", "middle", "α=.10, q=.01, p=.10\nr=1; middle"),
    ("rescue_late", "s_staged_al10_q01_e45_kh0_p10_sim01", "late", "Same task\nr=1; late"),
    ("reversal_middle", "s_staged_al10_q03_e45_kh0_p10_sim02", "middle", "α=.10, q=.03, p=.10\nr=2; middle"),
)
COLORS = {"hold": "#666666", "switch": "#2171b5", "bridge": "#238b45", "hide_reposts": "#d95f0e"}
CASE_ARMS = ("hold", "switch", "bridge")
CASE_STYLES = {
    "hold": {"ls": (0, (3, 3)), "marker": "s", "markevery": (0, 6), "lw": 1.2, "zorder": 5},
    "switch": {"ls": "-", "marker": "o", "markevery": (2, 6), "lw": 1.6, "zorder": 3},
    "bridge": {"ls": (0, (5, 2, 1, 2)), "marker": "^", "markevery": (4, 6), "lw": 1.3, "zorder": 4},
}
INSERTED_COLOR = "#2171b5"
ZERO_INSERTION_COLOR = "#777777"


def _compress_csv(path):
    with path.open("rb") as source, path.with_suffix(".csv.gz").open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as stream:
            shutil.copyfileobj(source, stream)


def _recover_data(output):
    current, backup = output / "data", output / ".previous-data"
    if not backup.exists():
        return
    if current.exists():
        manifest = json.loads((current / "manifest.json").read_text())
        if manifest["schema"] != SCHEMA or any(sha(current / name) != expected for name, expected in manifest["files"].items()):
            raise ValueError("interrupted portable-data replacement requires inspection")
        shutil.rmtree(backup)
    else:
        backup.rename(current)


def export_bundle(dataset, output):
    """Copy authenticated completed summaries, then calculate local figure tables."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("the portable bundle must be separate from the source dataset")
    plan, records = read_study(dataset, "staged")
    if not records:
        raise ValueError("no completed staged records")
    output.mkdir(parents=True, exist_ok=True)
    _recover_data(output)
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
    try:
        study = stage / "study"
        (study / "records").mkdir(parents=True)
        shutil.copy2(dataset / "plan.json", study / "plan.json")
        for identifier in records:
            shutil.copy2(dataset / "records" / f"{identifier}.json.gz", study / "records" / f"{identifier}.json.gz")
        for relative in ("snapshot.json", "runtime/binding.json"):
            source = dataset / relative
            if source.is_file():
                (study / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, study / relative)
        tables = stage / "tables"
        analyze(study, tables, make_figures=False)
        for path in tables.glob("*.csv"):
            _compress_csv(path)
            path.unlink()
        manifest = {
            "schema": SCHEMA, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset), "plan_fingerprint": plan["plan_fingerprint"],
            "completed": len(records), "planned": len(plan["tasks"]),
            "files": {str(path.relative_to(stage)): sha(path) for path in sorted(stage.rglob("*")) if path.is_file()},
        }
        atomic_json(stage / "manifest.json", manifest)
        backup = output / ".previous-data"
        current = output / "data"
        if current.exists():
            current.rename(backup)
        try:
            stage.rename(current)
        except BaseException:
            if backup.exists():
                backup.rename(current)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def _bundle_manifest(bundle):
    _recover_data(bundle)
    manifest = json.loads((bundle / "data/manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported manuscript data schema")
    for relative, expected in manifest["files"].items():
        path = bundle / "data" / relative
        if not path.resolve().is_relative_to((bundle / "data").resolve()) or sha(path) != expected:
            raise ValueError(f"portable data changed: {relative}")
    return manifest


def paired_effects(runs, arm, control):
    """Join within task and checkpoint, rejecting ambiguous or mismatched pairs."""
    selected = runs[runs["arm"].isin((arm, control))]
    keys = ["task_id", "stage"]
    if selected.duplicated([*keys, "arm"]).any():
        raise ValueError("duplicate task/stage/arm observations")
    left, right = (selected[selected["arm"] == name].set_index(keys) for name in (arm, control))
    joined = left.join(right, how="inner", lsuffix="_arm", rsuffix="_control")
    for name in ("start_step", "start_separation", "start_excess"):
        a, b = joined[name + "_arm"], joined[name + "_control"]
        if not ((a == b) | (a.isna() & b.isna())).all():
            raise ValueError(f"paired branches have different {name}")
    result = joined[[name + "_arm" for name in (
        "setting", "sensitivity", "influence", "rewiring", "repost", "replicate", "epsilon", "history",
        "bridge_shown", "cross_confident", "updates",
    )]].rename(columns=lambda name: name.removesuffix("_arm"))
    for name in ("separation", "excess"):
        result["delta_" + name] = joined["end_" + name + "_arm"] - joined["end_" + name + "_control"]
    return result.reset_index()


def _rows(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def panel_specifications(bundle):
    manifest = _bundle_manifest(bundle)
    plan, records = read_study(bundle / "data/study", "staged")
    thresholds = {"separation": plan["protocol"]["separation_threshold"],
                  "excess_concordance": plan["protocol"]["excess_threshold"]}
    runs = pd.read_csv(bundle / "data/tables/runs.csv.gz", float_precision="round_trip")
    core = runs[runs["sensitivity"] == "core"]
    native, bridge = paired_effects(core, "switch", "hold"), paired_effects(core, "bridge", "switch")
    settings = sorted({
        (task["metadata"]["HKParams"]["Influence"], task["metadata"]["HKParams"]["RewiringRate"])
        for task in plan["tasks"] if task["sensitivity"] == "core"
    })
    panels = {}
    for repost in sorted(core["repost"].unique()):
        panels[f"policy_p{round(100 * repost):02}"] = {
            "kind": "policy", "title": f"p={repost:g}", "settings": settings,
            "rows": _rows(native[native["repost"] == repost]),
        }
    for index, (influence, rewiring) in enumerate(CROSS_SETTINGS):
        selected = bridge[(bridge["influence"] == influence) & (bridge["rewiring"] == rewiring)]
        for metric in ("delta_separation", "delta_excess", "dose"):
            panels[f"bridge_{index}_{metric}"] = {
                "kind": "bridge", "title": f"α={influence:g}, q={rewiring:g}", "metric": metric,
                "rows": _rows(selected),
            }
    for name, identifier, phase, title in CASES:
        if identifier not in records or phase not in records[identifier]["stages"]:
            raise ValueError(f"illustrative case unavailable: {identifier}/{phase}")
        block = records[identifier]["stages"][phase]
        for metric in ("separation", "excess_concordance"):
            panels[f"case_{name}_{metric}"] = {
                "kind": "case", "title": title, "metric": metric, "task_id": identifier,
                "threshold": thresholds[metric],
                "checkpoint": block["checkpoint"]["step"],
                "start_excess": block["checkpoint"]["pathway"]["excess_concordance"],
                "arms": {arm: {"step": [point["step"] for point in data["endpoints"]],
                               "value": [point["pathway"][metric] for point in data["endpoints"]]}
                         for arm, data in block["arms"].items()},
            }
    coverage = coverage_by_setting(plan, records)
    return {
        "schema": SCHEMA, "data_manifest_sha256": sha(bundle / "data/manifest.json"),
        "completed": manifest["completed"], "planned": manifest["planned"],
        "core_completed": len(set(core["task_id"])), "core_planned": sum(t["sensitivity"] == "core" for t in plan["tasks"]),
        "source_dataset": manifest["source_dataset"], "plan_fingerprint": plan["plan_fingerprint"],
        "coverage": coverage, "panels": panels,
        "thresholds": thresholds,
        "comparison_unit": "paired arms within one task and checkpoint; parameter cells have independent initial states",
    }


def default_layout(specs):
    return {
        "schema": SCHEMA,
        "style": {"font_size": 8, "marker_size": 3.3, "colors": COLORS, "heatmap_limit": 1.0,
                  "bridge_detail_columns": [0, 1]},
        "figures": {
            "staged_policy_effects": {"width": 7.1, "height": 4.4, "rows": 1, "columns": 3,
                "left": 0.14, "right": 0.87, "bottom": 0.08, "top": 0.87, "hspace": 0.3, "wspace": 0.08,
                "panels": [name for name in specs["panels"] if name.startswith("policy_")]},
            "staged_bridge_effects": {"width": 7.1, "height": 5.9, "rows": 3, "columns": 4,
                "left": 0.085, "right": 0.985, "bottom": 0.16, "top": 0.87, "hspace": 0.6, "wspace": 0.35,
                "panels": [f"bridge_{index}_{metric}" for metric in ("delta_separation", "delta_excess", "dose") for index in range(4)]},
            "staged_intervention_cases": {"width": 7.1, "height": 4.9, "rows": 2, "columns": 3,
                "left": 0.085, "right": 0.985, "bottom": 0.18, "top": 0.82, "hspace": 0.43, "wspace": 0.22,
                "panels": [f"case_{name}_{metric}" for metric in ("separation", "excess_concordance") for name, *_ in CASES]},
        },
    }


def _decorate(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)


def _bridge_limits(specs):
    limits = {}
    for metric in ("delta_separation", "delta_excess"):
        values = [row[metric] for panel in specs["panels"].values()
                  if panel["kind"] == "bridge" and panel["metric"] == metric for row in panel["rows"]]
        low, high = min([0, *values]), max([0, *values])
        padding = max(0.02, (high - low) * 0.15)
        limits[metric] = low - padding, high + padding
    rates = [1000 * row["bridge_shown"] / row["updates"]
             for panel in specs["panels"].values() if panel["kind"] == "bridge" and panel["metric"] == "dose"
             for row in panel["rows"]]
    largest = max([0, *rates])
    limits["dose"] = -max(1, largest * 0.035), max(5, largest * 1.2)
    return limits


def _case_limits(specs):
    values = [value for panel in specs["panels"].values()
              if panel["kind"] == "case" and panel["metric"] == "excess_concordance"
              for data in panel["arms"].values() for value in data["value"]]
    thresholds = [panel["threshold"] for panel in specs["panels"].values()
                  if panel["kind"] == "case" and panel["metric"] == "excess_concordance"]
    low, high = min([0, *values, *thresholds]), max([0, *values, *thresholds])
    padding = max(0.01, (high - low) * 0.08)
    return {"separation": (-0.025, 1.025), "excess_concordance": (low - padding, high + padding)}


def _figure_style(specs, layout):
    panel_limits = {}
    detail_columns = layout["style"].get("bridge_detail_columns", [0, 1])
    for name, panel in specs["panels"].items():
        if (panel["kind"] == "bridge" and panel["metric"] != "dose"
                and any(name.startswith(f"bridge_{column}_") for column in detail_columns)):
            values = [row[panel["metric"]] for row in panel["rows"]]
            low, high = min([0, *values]), max([0, *values])
            padding = max(0.005, (high - low) * 0.15)
            panel_limits[(panel["title"], panel["metric"])] = low - padding, high + padding
    return {**layout["style"], "bridge_limits": _bridge_limits(specs), "case_limits": _case_limits(specs),
            "bridge_panel_limits": panel_limits}


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
    arms = spec["arms"]
    if not all(arm in arms for arm in CASE_ARMS):
        return None
    hold, switch, bridge = (arms[arm] for arm in CASE_ARMS)
    if hold != switch or bridge["step"] != hold["step"]:
        return None
    equal = np.asarray(bridge["value"]) == np.asarray(hold["value"])
    first_different = np.flatnonzero(~equal)
    if not len(first_different):
        return "Hold = switch = bridge\nat all recorded steps"
    if first_different[0] == 0:
        return "Hold = switch\nat all recorded steps"
    last_shared = hold["step"][first_different[0] - 1]
    return f"Hold = switch throughout\nAll coincide through recorded\nstep t = {last_shared}"


def _draw_panel(fig, ax, spec, style, *, compact=False):
    kind = spec["kind"]
    if kind == "policy":
        matrix = np.full((len(spec["settings"]), 3), np.nan)
        counts = np.zeros_like(matrix, dtype=int)
        rows = pd.DataFrame(spec["rows"])
        for i, (influence, rewiring) in enumerate(spec["settings"]):
            cell = rows[(rows["influence"] == influence) & (rows["rewiring"] == rewiring)]
            for j, phase in enumerate(("early", "middle", "late")):
                values = cell.loc[cell["stage"] == phase, "delta_separation"].dropna()
                if not values.empty:
                    matrix[i, j], counts[i, j] = values.mean(), len(values)
        limit = style["heatmap_limit"]
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            text = f"{value:+.3f}\nn={counts[i,j]}" if np.isfinite(value) else "unavailable"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 * limit else "black")
        ax.set(xticks=range(3), xticklabels=("Early", "Middle", "Late"), yticks=range(len(matrix)),
               yticklabels=[f"({a:g}, {q:g})" for a, q in spec["settings"]], ylabel="(α, q)")
        if not compact:
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, label="ΔP: switch − hold")
    elif kind == "bridge":
        rows = pd.DataFrame(spec["rows"])
        metric = spec["metric"]
        ax.set(xlim=(-0.45, 2.48), xticks=range(3), xticklabels=("0", ".10", ".25"), xlabel="Repost probability p",
               ylabel={"delta_separation": "ΔP: bridge − switch", "delta_excess": "ΔE: bridge − switch",
                       "dose": "Bridge slots\n/ 1,000 updates"}[metric])
        ax.set_ylim(style.get("bridge_panel_limits", {}).get((spec["title"], metric), style["bridge_limits"][metric]))
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
            positions, size = _swarm_positions(ax, j - 0.08, [p[0] for p in points], style["marker_size"])
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
        metric = spec["metric"]
        for arm in CASE_ARMS:
            if arm not in spec["arms"]:
                continue
            data = spec["arms"][arm]
            ax.plot(data["step"], data["value"], label=label(arm), color=style["colors"][arm],
                    **CASE_STYLES[arm], markersize=2.8, markeredgewidth=0.65,
                    markerfacecolor="white" if arm == "hold" else style["colors"][arm])
        ax.axhline(spec["threshold"], color="0.6", ls=":", lw=0.8)
        ax.set(xlabel="Simulation step", ylabel="Opinion separation P" if metric == "separation" else "Excess concordance E")
        ax.set_ylim(style["case_limits"][metric])
        note = _case_overlap_note(spec)
        if note:
            ax.text(0.035, 0.06 if metric == "separation" else 0.94, note, transform=ax.transAxes,
                    fontsize=6, va="bottom" if metric == "separation" else "top", linespacing=1.3,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
        if not compact:
            ax.legend(fontsize=7)
        _decorate(ax)
    ax.set_title(spec["title"], fontsize=8)


def _save_panel(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    save(fig, path)


def prepare(bundle, output):
    """Save self-contained plot data and an editable layout, without panel images."""
    bundle, output = Path(bundle), Path(output)
    specs = panel_specifications(bundle)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plot_data.json", specs)
    layout_path = output / "layout.json"
    if not layout_path.exists():
        atomic_json(layout_path, default_layout(specs))
    return specs


def render_panels(bundle, output):
    """Optionally export independent SVG/PDF/PNG alongside the compact plot data."""
    bundle, output = Path(bundle), Path(output)
    specs = prepare(bundle, output)
    atomic_json(output / "panels/specs.json", specs)
    layout_path = output / "layout.json"
    style = _figure_style(specs, json.loads(layout_path.read_text()))
    setup_paper_params()
    plt.rcParams.update({"font.size": style["font_size"], "svg.fonttype": "none"})
    for name, spec in specs["panels"].items():
        fig, ax = plt.subplots(figsize=(4.0, 4.8 if spec["kind"] == "policy" else 2.8), layout="constrained")
        _draw_panel(fig, ax, spec, style)
        _save_panel(fig, output / "panels" / name)
    atomic_json(output / "panels/manifest.json", {
        "schema": SCHEMA, "data_manifest_sha256": specs["data_manifest_sha256"],
        "specifications_sha256": sha(output / "panels/specs.json"), "renderer_sha256": sha(__file__),
        "portable_bundle": str(bundle.resolve()),
        "files": {p.name: sha(p) for p in sorted((output / "panels").iterdir()) if p.suffix in (".svg", ".pdf", ".png")},
    })
    return specs


def compose(directory):
    """Compose vector figures from saved panel data; no dataset or bundle access."""
    directory = Path(directory)
    data_path = directory / "plot_data.json"
    if not data_path.exists():
        data_path = directory / "panels/specs.json"  # Older exported directories.
    specs = json.loads(data_path.read_text())
    layout = json.loads((directory / "layout.json").read_text())
    if specs["schema"] != SCHEMA or layout["schema"] != SCHEMA:
        raise ValueError("unsupported panel or layout schema")
    style = _figure_style(specs, layout)
    setup_paper_params()
    plt.rcParams.update({"font.size": style["font_size"], "axes.labelsize": style["font_size"],
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    prefix = "Preliminary: " if specs["completed"] < specs["planned"] else ""
    titles = {
        "staged_policy_effects": "Mean paired switching effects",
        "staged_bridge_effects": "Bridge effects and actual insertion",
        "staged_intervention_cases": "Paired intervention responses",
    }
    for name, config in layout["figures"].items():
        fig, axes = plt.subplots(config["rows"], config["columns"], figsize=(config["width"], config["height"]), squeeze=False)
        fig.subplots_adjust(**{key: config[key] for key in ("left", "right", "bottom", "top", "hspace", "wspace")})
        for index, ax in enumerate(axes.flat):
            if index >= len(config["panels"]):
                ax.set_visible(False)
                continue
            panel_name = config["panels"][index]
            _draw_panel(fig, ax, specs["panels"][panel_name], style, compact=True)
            title = specs["panels"][panel_name]["title"]
            ax.set_title("")
            if name == "staged_intervention_cases":
                title = f"{title}; t={specs['panels'][panel_name]['checkpoint']}" if index < 3 else ""
            ax.set_title(f"({chr(97 + index)}) {title}", loc="left", fontsize=8)
            if index % config["columns"]:
                ax.set_ylabel("")
                if name != "staged_bridge_effects":
                    ax.tick_params(axis="y", labelleft=False)
            if name != "staged_policy_effects" and index < (config["rows"] - 1) * config["columns"]:
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
            handles = [Line2D([], [], color=style["colors"][arm],
                              **{key: value for key, value in CASE_STYLES[arm].items() if key != "markevery"},
                              markersize=3, markerfacecolor="white" if arm == "hold" else style["colors"][arm],
                              label=label(arm)) for arm in CASE_ARMS]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=7)
        _save_panel(fig, directory / "figures" / name)
    atomic_json(directory / "figure_manifest.json", {
        "schema": SCHEMA, "completed": specs["completed"], "planned": specs["planned"],
        "data_manifest_sha256": specs["data_manifest_sha256"],
        "plot_data": str(data_path.relative_to(directory)), "plot_data_sha256": sha(data_path),
        "panel_specifications_sha256": sha(data_path), "layout_sha256": sha(directory / "layout.json"),
        "renderer_sha256": sha(__file__), "source_dataset": specs["source_dataset"],
        "plan_fingerprint": specs["plan_fingerprint"],
        "files": {p.name: sha(p) for p in sorted((directory / "figures").iterdir()) if p.suffix in (".svg", ".pdf", ".png")},
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Freeze completed summaries into a local portable bundle")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    preparation = commands.add_parser("prepare", help="Save compact plot data and layout without independent panel files")
    preparation.add_argument("--bundle", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    panels = commands.add_parser("panels", help="Optionally save independent editable panel files")
    panels.add_argument("--bundle", type=Path, required=True)
    panels.add_argument("--output", type=Path, required=True)
    composition = commands.add_parser("compose", help="Compose figures from panel data and layout only")
    composition.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export_bundle(args.dataset, args.output)
    elif args.command == "prepare":
        prepare(args.bundle, args.output)
    elif args.command == "panels":
        render_panels(args.bundle, args.output)
    else:
        compose(args.directory)


if __name__ == "__main__":
    main()
