"""Render original-scan heatmaps from one explicitly selected analysis snapshot."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

import experiments.paper.scenarios as cfg
from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import read_statistics, select_statistics
from experiments.paper.paths import MAIN_SCAN_OUTPUT, statistics_database


@dataclass(frozen=True, slots=True)
class ColorScale:
    cmap: str
    limits: tuple[float, float]
    ticks: tuple[float, ...]


IW_SCALE = ColorScale("RdYlBu_r", (0.0, 1.0), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
IW_DIFFERENCE_SCALE = ColorScale("RdBu_r", (-0.6, 0.6), (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6))
TV_SCALE = ColorScale("YlGnBu", (0.5, 2.0), (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0))
TV_DIFFERENCE_SCALE = ColorScale("RdBu_r", (-1.0, 1.0), (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0))
ACTIVE_STEP_SCALE = ColorScale("YlGnBu", (1.0, 5.0), (1.0, 2.0, 3.0, 4.0, 5.0))


@dataclass(frozen=True, slots=True)
class HeatmapMetric:
    measure: Callable[[ScenarioStatistics], float]
    label: str
    color_scale: ColorScale
    difference_scale: ColorScale | None = None
    peaks: int | None = None


@dataclass(frozen=True, slots=True)
class HeatmapData:
    metric: HeatmapMetric
    structure_mean: NDArray[np.float64]
    structure_std: NDArray[np.float64]
    opinion_mean: NDArray[np.float64]
    opinion_std: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CaptionGroup:
    label: str
    count: int
    scale: ColorScale
    below_scale: int
    above_scale: int


@dataclass(frozen=True, slots=True)
class HeatmapCaption:
    analysis_id: str
    origin: str
    available_completed: int
    groups: tuple[CaptionGroup, ...]

    def write(self, path: Path):
        lines = [
            "Original-scan heatmaps from the selected completed trajectories.",
            f"analysis_id: {self.analysis_id}",
            f"origin: {self.origin}",
            f"Available completed trajectories in the selected snapshot: n={self.available_completed}.",
            *[f"{group.label}: n={group.count}; fixed {group.scale.cmap} scale {group.scale.limits}; "
              f"individual values below/above this scale: {group.below_scale}/{group.above_scale}."
              for group in self.groups],
            "Counts refer to finite observations after each metric's filters; these counts do not imply a complete scan.",
            "Each heatmap cell is the mean of the matching trajectories. Cells without matching finite values are blank, not zero.",
            "Difference panels, where present, show opinion minus structure only where both cells have observations.",
            "Color maps, limits and ticks follow the existing paper figures. Values beyond a displayed color scale saturate at its endpoints; the underlying values remain unchanged.",
            "All heatmaps restrict retained-post history to k=0. For the TV(I_p) panels, TV is the observed sum of absolute successive changes in I_p; #P=1 denotes one final opinion peak.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


rw_rev_dict = {value: index for index, value in enumerate(cfg.REWIRING_AXIS)}
dc_rev_dict = {value: index for index, value in enumerate(cfg.INFLUENCE_AXIS)}
RETWEET_GROUPS = ((0.0, "StructureM9"), (0.0, "OpinionM9"), (0.5, "StructureM9"), (0.5, "OpinionM9"))


def _create_evaluator(full_data_selector: Iterable[ScenarioStatistics]):
    cells: list[list[list[ScenarioStatistics]]] = [
        [[] for _ in cfg.INFLUENCE_AXIS] for _ in cfg.REWIRING_AXIS
    ]
    for datum in full_data_selector:
        cells[rw_rev_dict[datum.rewiring]][dc_rev_dict[datum.decay]].append(datum)

    def evaluate(measure: Callable[[ScenarioStatistics], float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        means = np.full((len(cfg.REWIRING_AXIS), len(cfg.INFLUENCE_AXIS)), np.nan)
        deviations = np.full_like(means, np.nan)
        for row_index, row in enumerate(cells):
            for column_index, cell in enumerate(row):
                values = np.asarray([measure(record) for record in cell], dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    means[row_index, column_index] = values.mean()
                    deviations[row_index, column_index] = values.std()
        return means, deviations

    return evaluate


def _heatmap_rows(session: Iterable[ScenarioStatistics], rs_key: str, opinion_peak_count: int | None = None):
    return select_statistics(
        session,
        *((lambda row: row.last_opinion_peak_count == opinion_peak_count,) if opinion_peak_count is not None else ()),
        lambda row: row.name.startswith("s_grad"),
        lambda row: row.recsys_type == rs_key,
        lambda row: row.post_retain_count == 0,
        lambda row: row.pathway_integral is not None,
    )


def create_heatmap_evaluator(session: Iterable[ScenarioStatistics], rs_key: str, opinion_peak_count: int | None = None):
    return _create_evaluator(_heatmap_rows(session, rs_key, opinion_peak_count))


def _retweet_rows(
    session: Iterable[ScenarioStatistics], retweet_rate: float, rs_key: str | None = None
):
    return select_statistics(
        session,
        lambda row: row.name.startswith("s_grad"),
        *((lambda row: row.recsys_type == rs_key,) if rs_key else ()),
        lambda row: row.repost_rate == retweet_rate,
        lambda row: row.post_retain_count == 0,
        lambda row: row.pathway_integral is not None,
    )


def create_heatmap_evaluator_retweet(
    session: Iterable[ScenarioStatistics], retweet_rate: float, rs_key: str | None = None
):
    return _create_evaluator(_retweet_rows(session, retweet_rate, rs_key))


def write_heatmap_caption(
    path: Path, session: Sequence[ScenarioStatistics], metrics: Sequence[HeatmapMetric], *, by_retweet=False
):
    groups = []
    for metric in metrics:
        if by_retweet:
            selections = ((f"{strategy}, p={repost:g}", _retweet_rows(session, repost, strategy))
                          for repost, strategy in RETWEET_GROUPS)
        else:
            selections = ((strategy, _heatmap_rows(session, strategy, metric.peaks))
                          for strategy in ("StructureM9", "OpinionM9"))
        for label, rows in selections:
            values = np.asarray([metric.measure(row) for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            lower, upper = metric.color_scale.limits
            groups.append(CaptionGroup(f"{metric.label} ({label})", int(values.size), metric.color_scale,
                                       int(np.count_nonzero(values < lower)), int(np.count_nonzero(values > upper))))
    HeatmapCaption(session[0].analysis_id, session[0].origin, len(session), tuple(groups)).write(path)


def _plot_single_heatmap(ax, data_mean, data_std, title, scale: ColorScale, with_std=True):
    im = ax.imshow(data_mean, cmap=scale.cmap, vmin=scale.limits[0], vmax=scale.limits[1])
    ax.set_title(title, loc="left")
    if with_std:
        for i, row in enumerate(data_std):
            for j, value in enumerate(row):
                if np.isfinite(value):
                    ax.scatter(j, i, s=value * 1000, facecolors="none", edgecolors="midnightblue", linewidths=2, alpha=0.6)
    return im


def _setup_heatmap_axes(axes: Sequence[Axes], *, n_rows: int, n_columns: int):
    for index, axis in enumerate(axes):
        axis.invert_yaxis()
        axis.set_xticks(np.arange(len(cfg.INFLUENCE_AXIS)))
        axis.set_yticks(np.arange(len(cfg.REWIRING_AXIS)))
        if index // n_columns == n_rows - 1:
            axis.set_xlabel(r"influence ($\alpha$)")
            axis.set_xticklabels(cfg.INFLUENCE_AXIS, rotation=90)
        else:
            axis.set_xticklabels([])
        if index % n_columns == 0:
            axis.set_ylabel(r"rewiring ($q$)")
            axis.set_yticklabels(cfg.REWIRING_AXIS)
        else:
            axis.set_yticklabels([])


def plot_combined_heatmaps(
    session: Sequence[ScenarioStatistics], metrics: Sequence[HeatmapMetric], *, with_diff=True, with_std=True
):
    if not metrics:
        raise ValueError("at least one heatmap metric is required")
    if with_diff and any(metric.difference_scale is None for metric in metrics):
        raise ValueError("difference heatmaps require an explicit color scale")
    all_data: list[HeatmapData] = []
    for metric in metrics:
        mean_st, std_st = create_heatmap_evaluator(session, "StructureM9", metric.peaks)(metric.measure)
        mean_op, std_op = create_heatmap_evaluator(session, "OpinionM9", metric.peaks)(metric.measure)
        all_data.append(HeatmapData(metric, mean_st, std_st, mean_op, std_op))

    n_columns = 3 if with_diff else 2
    fig, axes_orig = plt_figure(n_row=len(metrics), n_col=n_columns, hw_ratio=1, constrained_layout=False)
    axes = np.asarray(axes_orig).ravel().tolist()
    for metric_index, data in enumerate(all_data):
        base_index = metric_index * n_columns
        for offset, (mean, std, strategy) in enumerate((
            (data.structure_mean, data.structure_std, "structure"),
            (data.opinion_mean, data.opinion_std, "opinion"),
        )):
            axis = axes[base_index + offset]
            label = chr(ord("a") + base_index + offset)
            image = _plot_single_heatmap(axis, mean, std, f"({label}) {data.metric.label} ({strategy})",
                                         data.metric.color_scale, with_std)
            fig.colorbar(image, ax=axis, ticks=data.metric.color_scale.ticks, fraction=0.046, pad=0.04)
        if with_diff:
            difference = data.opinion_mean - data.structure_mean
            scale = data.metric.difference_scale
            assert scale is not None
            axis = axes[base_index + 2]
            label = chr(ord("a") + base_index + 2)
            image = _plot_single_heatmap(axis, difference, np.zeros_like(difference),
                                         f"({label}) difference (opinion - structure)", scale, False)
            fig.colorbar(image, ax=axis, ticks=scale.ticks, fraction=0.046, pad=0.04)
    _setup_heatmap_axes(axes, n_rows=len(metrics), n_columns=n_columns)
    fig.tight_layout()
    return fig


def plot_heatmap_by_retweet(session: Sequence[ScenarioStatistics], metric: HeatmapMetric):
    statistics = [create_heatmap_evaluator_retweet(session, repost, strategy)(metric.measure)
                  for repost, strategy in RETWEET_GROUPS]
    fig, axes = plt_figure(n_row=1, n_col=len(RETWEET_GROUPS), hw_ratio=1, constrained_layout=False)
    for index, (axis, (mean, std), (repost, strategy)) in enumerate(zip(axes, statistics, RETWEET_GROUPS)):
        label = chr(ord("a") + index)
        short_strategy = "St." if strategy == "StructureM9" else "Op."
        image = _plot_single_heatmap(axis, mean, std, f"({label}) {metric.label} ({short_strategy}, p={repost:g})",
                                     metric.color_scale, False)
        fig.colorbar(image, ax=axis, ticks=metric.color_scale.ticks, fraction=0.046, pad=0.04)
    _setup_heatmap_axes(axes, n_rows=1, n_columns=len(RETWEET_GROUPS))
    fig.tight_layout()
    return fig


def _format_summary(label: str, values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return f"{label}: n=0; no matching finite observations"
    return (f"{label}: n={array.size}, mean={array.mean():.4f}, std={array.std():.4f}, "
            f"min={array.min():.4f}, max={array.max():.4f}")


def main():
    setup_paper_params()
    session = read_statistics(statistics_database(), mode="gradation")
    figure_dir = MAIN_SCAN_OUTPUT.resolve() / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    def pathway(row: ScenarioStatistics) -> float:
        assert row.pathway_integral is not None
        return row.pathway_integral

    def total_variation(row: ScenarioStatistics) -> float:
        assert row.p_backdrop is not None
        return row.p_backdrop

    def active_step(row: ScenarioStatistics) -> float:
        assert row.active_step is not None
        return float(np.log10(max(row.active_step, 1)))

    metrics = [HeatmapMetric(pathway, "$I_w$", IW_SCALE, IW_DIFFERENCE_SCALE),
               HeatmapMetric(total_variation, r"TV($I_p$), #P=1", TV_SCALE, TV_DIFFERENCE_SCALE, peaks=1)]
    plt_save_and_close(plot_combined_heatmaps(session, metrics, with_std=False),
                       str(figure_dir / "f_grad_and_pol_trad_index"))
    write_heatmap_caption(figure_dir / "f_grad_and_pol_trad_index_caption.txt", session, metrics)

    summaries = []
    for strategy, label in (("OpinionM9", "p_index_op"), ("StructureM9", "p_index_st")):
        rows = select_statistics(session, lambda row: row.name.startswith("s_grad"),
                                 lambda row: row.recsys_type == strategy,
                                 lambda row: row.post_retain_count == 0,
                                 lambda row: row.last_opinion_peak_count == 1,
                                 lambda row: row.pathway_integral is not None)
        summaries.append(_format_summary(label, (row.p_index[-1] for row in rows
                                                if row.p_index is not None and row.p_index.size)))
    for summary in summaries:
        print(summary)
    (figure_dir / "f_p_backdrop_stats.txt").write_text("\n".join(summaries) + "\n", encoding="utf-8")
    active_metric = HeatmapMetric(active_step, r"$\log_{10}(t_a)$", ACTIVE_STEP_SCALE)
    plt_save_and_close(plot_heatmap_by_retweet(session, active_metric),
                       str(figure_dir / "f_p_active_step"))
    write_heatmap_caption(figure_dir / "f_p_active_step_caption.txt", session, [active_metric], by_retweet=True)


if __name__ == "__main__":
    main()
