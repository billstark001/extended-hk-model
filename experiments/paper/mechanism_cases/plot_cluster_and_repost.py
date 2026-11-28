"""Render the reposting comparison for an explicit saved-run history design."""

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
from matplotlib.axes import Axes

from common.numerics import adaptive_moving_stats, get_kde_pdf
from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import (
    descriptive_iw_cutoff,
    read_statistics,
    select_statistics,
)
from experiments.paper.paths import MECHANISM_OUTPUT, statistics_database
from experiments.paper.pathway_index.grouped_summary import HISTORY_SCOPES, history_in_scope


def draw_prob(
    filter_1: np.ndarray,
    filter_2: np.ndarray,
    filter_3: np.ndarray,
    total_size: int,
    ax: Axes,
    stats_grad: np.ndarray,
    x_axis: np.ndarray,
):
    if total_size == 0:
        return

    def _f(mask: np.ndarray):
        data_x = stats_grad[mask]
        if data_x.size == 0:
            return x_axis * 0
        return get_kde_pdf(data_x, 0.025, x_axis[0], x_axis[-1])(x_axis)

    div_ratio_1 = np.sum(filter_1) / total_size
    div_ratio_2 = np.sum(filter_2) / total_size
    div_ratio_3 = np.sum(filter_3) / total_size

    div_group_1 = _f(filter_1) * div_ratio_1
    div_group_2 = _f(filter_2) * div_ratio_2
    div_group_3 = _f(filter_3) * div_ratio_3

    ax.fill_between(x_axis, x_axis * 0, div_group_1, label="#peaks = 1")
    ax.fill_between(x_axis, div_group_1, div_group_1 + div_group_2, label="#peaks = 2")
    ax.fill_between(x_axis, div_group_1 + div_group_2, div_group_1 + div_group_2 + div_group_3, label="#peaks >= 3")


def plot_bar(
    ax: Axes,
    all_data: List[List[Tuple[float, float]]] | np.ndarray,
    k_labels: List[str],
    m_labels: List[str],
    m_colors: List[Any],
):
    all_data_arr = np.array(all_data, dtype=float)  # (k, m, [mean, std])
    k, m, _ = all_data_arr.shape
    means = all_data_arr[:, :, 0]
    stds = all_data_arr[:, :, 1]

    width = 0.6 / m  # width of the bars
    x = np.arange(k)

    for i in range(m):
        available = np.isfinite(means[:, i]) & np.isfinite(stds[:, i])
        ax.bar(
            (x + i * width)[available],
            means[:, i][available],
            width,
            yerr=stds[:, i][available],
            label=m_labels[i],
            capsize=4,
            edgecolor="black",
            color=m_colors[i],
            alpha=0.5,
        )

    ax.set_xticks(x + width * (m - 1) / 2)
    ax.set_xticklabels(k_labels)
    fit_uncertainty_limits(ax, means, stds)

    return ax


def fit_uncertainty_limits(ax, means, stds):
    """Retain signed observations and the complete mean-plus/minus-SD range."""
    means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
    available = np.isfinite(means) & np.isfinite(stds)
    if not np.any(available):
        return
    lower = min(0., float(np.min((means - stds)[available])))
    upper = max(0., float(np.max((means + stds)[available])))
    padding = .05 * (upper - lower) if upper > lower else .05
    ax.set_ylim(lower - padding, upper + padding)


def pooled_summary(records):
    """Keep the existing equal-run mixture mean and within/between variance."""
    if not records:
        return np.full(5, np.nan), np.full(5, np.nan)
    raw_mean, raw_std = np.array(
        [(row.opinion_diff_seg_mean, row.opinion_diff_seg_std) for row in records]
    ).transpose(1, 0, -1)
    mean = np.mean(raw_mean, axis=0)
    variance = np.mean(raw_std**2 + (raw_mean - mean)**2, axis=0)
    return mean, variance**.5


def _moving_curve(records, reposting, x_min, x_max):
    rows = [row for row in records if (row.repost_rate != 0) == reposting]
    if not rows:
        return np.array([]), np.array([]), np.array([])
    x, mean, std = np.array([
        (row.pathway_integral, row.opinion_diff_seg_mean[1], row.opinion_diff_seg_std[1])
        for row in rows
    ]).T
    # Use identical support, smoothing and evaluation coordinates for the band.
    support = {} if reposting else {"min": x_min, "max": x_max}
    x_fit, mean_fit, _ = adaptive_moving_stats(x, mean, .2, **support)
    _, std_fit, _ = adaptive_moving_stats(x, std, .2, **support, result_x=x_fit)
    return x_fit, mean_fit, std_fit


def render_cluster_and_repost(records, cutoff, *, history_scope="original-design"):
    if not math.isfinite(cutoff):
        raise ValueError("cutoff must be finite")
    if history_scope not in HISTORY_SCOPES:
        raise ValueError(f"unknown history scope: {history_scope}")
    session = select_statistics(
        records, lambda row: row.name.startswith("s_grad"),
        lambda row: history_in_scope(row.recsys_type, row.post_retain_count, history_scope),
        lambda row: row.pathway_integral is not None and math.isfinite(row.pathway_integral),
    )
    if not session:
        raise ValueError(f"no finite gradation trajectories in history scope {history_scope}")

    stats_f1 = select_statistics(
        session, lambda row: row.last_opinion_peak_count is not None
    )
    if not stats_f1:
        raise ValueError("no terminal opinion-peak observations for the selected trajectories")

    stats_grad, stats_peak, stats_retweet = np.array(
        [(x.pathway_integral, x.last_opinion_peak_count, x.repost_rate) for x in stats_f1], dtype=np.float64
    ).T

    stats_peak = stats_peak.astype(int)

    x_min, x_max = float(stats_grad.min() - 0.1), float(stats_grad.max() + 0.1)
    x_axis = np.linspace(x_min, x_max, 400)

    stats_raw = select_statistics(
        session,
        lambda row: row.pathway_integral is not None,
        lambda row: row.opinion_diff_seg_mean is not None,
        lambda row: row.opinion_diff_seg_std is not None,
    )
    if not stats_raw:
        raise ValueError("no five-window opinion-difference statistics for the selected trajectories")
    for row in stats_raw:
        mean, std = np.asarray(row.opinion_diff_seg_mean), np.asarray(row.opinion_diff_seg_std)
        if mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError(f"invalid five-window opinion-difference statistics: {row.origin}/{row.name}")

    def _filter(f: Callable[[ScenarioStatistics], bool]):
        stats_filtered = [x for x in stats_raw if f(x)]

        return pooled_summary(stats_filtered)

    od_p2_r_mean, od_p2_r_std = _filter(lambda x: x.pathway_integral < cutoff and x.repost_rate != 0)
    od_p2_nr_mean, od_p2_nr_std = _filter(lambda x: x.pathway_integral < cutoff and x.repost_rate == 0)
    od_p1_r_mean, od_p1_r_std = _filter(lambda x: x.pathway_integral >= cutoff and x.repost_rate != 0)
    od_p1_nr_mean, od_p1_nr_std = _filter(lambda x: x.pathway_integral >= cutoff and x.repost_rate == 0)

    x1_f, y1_f, y1_std_f = _moving_curve(stats_raw, False, x_min, x_max)
    x2_f, y2_f, y2_std_f = _moving_curve(stats_raw, True, x_min, x_max)

    setup_paper_params()
    fig, axes = plt_figure(n_row=2, n_col=3)

    ax1, ax2, ax3 = axes[0]

    draw_prob(stats_peak == 1, stats_peak == 2, stats_peak > 2, stats_peak.size, ax1, stats_grad, x_axis)
    draw_prob(
        np.logical_and(stats_peak == 1, stats_retweet == 0),
        np.logical_and(stats_peak == 2, stats_retweet == 0),
        np.logical_and(stats_peak > 2, stats_retweet == 0),
        np.sum(stats_retweet == 0),
        ax2,
        stats_grad,
        x_axis,
    )
    draw_prob(
        np.logical_and(stats_peak == 1, stats_retweet != 0),
        np.logical_and(stats_peak == 2, stats_retweet != 0),
        np.logical_and(stats_peak > 2, stats_retweet != 0),
        np.sum(stats_retweet != 0),
        ax3,
        stats_grad,
        x_axis,
    )

    for ax in [ax1, ax2, ax3]:
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(r"$I_w$")

    ax1.set_ylabel(r"P(#peaks)")
    ax1.set_title("(a) gross", loc="left")
    ax2.set_title("(b) no retweet", loc="left")
    ax3.set_title("(c) with retweet", loc="left")

    ax4, ax5, ax6 = axes[1]

    x_labels = ["0~.2", ".2~.4", ".4~.6", ".6~.8", ".8~1"]

    plot_bar(
        ax4,
        np.array([[od_p2_nr_mean, od_p2_nr_std], [od_p2_r_mean, od_p2_r_std]]).transpose(2, 0, 1),
        x_labels,
        ["no retweet", "retweet"],
        ["tab:blue", "tab:cyan"],
    )

    plot_bar(
        ax5,
        np.array([[od_p1_nr_mean, od_p1_nr_std], [od_p1_r_mean, od_p1_r_std]]).transpose(2, 0, 1),
        x_labels,
        ["no retweet", "retweet"],
        ["tab:red", "tab:orange"],
    )

    ax6.plot(x1_f, y1_f, label="no retweet")
    ax6.fill_between(x1_f, y1_f - y1_std_f, y1_f + y1_std_f, alpha=0.1)
    ax6.plot(x2_f, y2_f, label="retweet")
    ax6.fill_between(x2_f, y2_f - y2_std_f, y2_f + y2_std_f, alpha=0.1)
    fit_uncertainty_limits(ax6, np.concatenate((y1_f, y2_f)), np.concatenate((y1_std_f, y2_std_f)))

    for ax in [ax4, ax5, ax6]:
        ax.grid(True)
        ax.legend()

    for ax in [ax4, ax5]:
        ax.set_xlabel(r"$t_n$")
    ax6.set_xlabel(r"$I_w$")

    ax4.set_ylabel("mean opinion difference")

    ax4.set_title(rf"(d) $I_w < {cutoff:g}$", loc="left")
    ax5.set_title(rf"(e) $I_w \geq {cutoff:g}$", loc="left")
    ax6.set_title(r"(f) mean op. diff. ($t_n \in [0.2, 0.4)$)", loc="left")

    return fig, session, stats_f1, stats_raw


def write_caption(stem, records, selected, density, windowed, cutoff, history_scope):
    origins = Counter(row.origin for row in selected)
    histories = Counter((row.recsys_type, row.post_retain_count) for row in selected)
    window_groups = Counter(("low" if row.pathway_integral < cutoff else "high",
                             "p=0" if row.repost_rate == 0 else "p>0") for row in windowed)
    lines = [
        f"Input completed trajectories: n={len(records)}; selected: n={len(selected)}.",
        f"History scope: {history_scope}. StructureM9 uses k=0; OpinionM9 uses "
        + ("k=0,2,6." if history_scope == "original-design" else "k=0."),
        f"Descriptive working groups: low I_w < {cutoff:g}; high I_w >= {cutoff:g}; not first-passage labels.",
        "Source counts: " + "; ".join(f"{name}: n={count}" for name, count in sorted(origins.items())),
        "Policy/history counts: " + "; ".join(f"{policy}/k={history}: n={count}" for (policy, history), count in sorted(histories.items())),
        f"Density observations with terminal peak counts: n={len(density)}; five-window observations: n={len(windowed)}.",
        "Five-window group counts: " + "; ".join(f"{group}/{repost}: n={count}" for (group, repost), count in sorted(window_groups.items())),
        "Every run has equal weight. Pooled policies are design mixtures over the declared histories and parameters; comparisons are not paired intervention effects.",
        "Density panels retain the existing Gaussian KDE and empirical peak-count weights. Lower bars retain equal-run means with within/between-run SD over the five saved normalized-time windows.",
        "The moving curve retains the existing adaptive smoothing of the second-window per-run means and SDs; its band is evaluated on the mean curve's grid and is descriptive, not a confidence interval.",
        "Y ranges include the observed mean plus/minus SD, including signed lower bounds; no stored value is clipped.",
        "Analysis identities: " + ", ".join(sorted({row.analysis_id for row in selected})),
    ]
    stem.with_name(stem.name + "_caption.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=statistics_database())
    parser.add_argument("--origin", action="append", dest="origins")
    parser.add_argument("--analysis-id")
    parser.add_argument("--output", type=Path, help="Output root; figures are saved below figures or zero/figures")
    parser.add_argument("--cutoff", type=float, help="Working split; otherwise require EHK_IW_CUTOFF")
    parser.add_argument("--history-scope", choices=HISTORY_SCOPES, default="original-design")
    args = parser.parse_args(argv)
    records = read_statistics(args.database, mode="gradation", analysis_id=args.analysis_id,
                              origins=tuple(args.origins) if args.origins else None)
    cutoff = descriptive_iw_cutoff() if args.cutoff is None else args.cutoff
    fig, selected, density, windowed = render_cluster_and_repost(records, cutoff, history_scope=args.history_scope)
    figure_dir = args.output if args.output is not None else MECHANISM_OUTPUT.resolve()
    if args.history_scope == "zero":
        figure_dir = figure_dir / "zero"
    figure_dir = figure_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "f_cluster_and_retweet"
    plt_save_and_close(fig, str(stem))
    write_caption(stem, records, selected, density, windowed, cutoff, args.history_scope)
    return {"runs": len(selected), "history_scope": args.history_scope, "cutoff": cutoff, "output": str(stem)}


if __name__ == "__main__":
    main()
