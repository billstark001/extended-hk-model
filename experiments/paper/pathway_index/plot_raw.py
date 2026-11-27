from functools import partial

import numpy as np

from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import descriptive_iw_cutoff, read_statistics
from experiments.paper.paths import PATHWAY_OUTPUT, statistics_database
from experiments.paper.pathway_index.bar_plot import PAPER_BAR_LIMITS, plot_grouped_bars, write_bar_caption
from experiments.paper.pathway_index.data_utils import piecewise_linear_integral_trapz

setup_paper_params()


k_filters_1 = [
    (lambda row: row.recsys_type == "StructureM9", lambda row: row.repost_rate == 0),
    (lambda row: row.recsys_type == "StructureM9", lambda row: row.repost_rate != 0),
    (lambda row: row.recsys_type == "OpinionM9", lambda row: row.repost_rate == 0),
    (lambda row: row.recsys_type == "OpinionM9", lambda row: row.repost_rate != 0),
]
k_filters_1 = [(*filters, lambda row: row.post_retain_count == 0) for filters in k_filters_1]

k_labels_1 = ["St, !R", "St, R", "Op, !R", "Op, R"]

k_filters_2 = [
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 0,
        lambda row: row.repost_rate == 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 2,
        lambda row: row.repost_rate == 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 6,
        lambda row: row.repost_rate == 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 0,
        lambda row: row.repost_rate != 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 2,
        lambda row: row.repost_rate != 0,
    ),
    (
        lambda row: row.recsys_type == "OpinionM9",
        lambda row: row.post_retain_count == 6,
        lambda row: row.repost_rate != 0,
    ),
]

k_labels_2 = ["k=0, !R", "k=2", "k=6", "k=0, R", "k=2", "k=6"]


if __name__ == "__main__":
    session = read_statistics(statistics_database(), mode="gradation")
    cutoff = descriptive_iw_cutoff()
    plot_matched = partial(plot_grouped_bars, records=session, cutoff=cutoff,
                           category_filters=k_filters_1, category_labels=k_labels_1)
    plot_history = partial(plot_grouped_bars, records=session, cutoff=cutoff,
                           category_filters=k_filters_2, category_labels=k_labels_2)

    fig, axes = plt_figure(n_row=2, n_col=3, constrained_layout=False)
    ax1, ax2 = axes[0][0], axes[0][1]
    ax3, ax4, ax5 = axes[1][0], axes[1][1], axes[1][2]

    axes[0][2].set_visible(False)

    def f_ev_count(x: ScenarioStatistics) -> float:
        assert x.event_count is not None
        return x.event_count

    def f_ev_step(x: ScenarioStatistics) -> float:
        assert x.active_step is not None
        return np.nan if x.event_step_mean is None else x.event_step_mean / max(x.active_step, 1)

    def f_auc_subj_pol_index(x: ScenarioStatistics) -> float:
        assert x.x_indices is not None
        assert x.active_step is not None
        assert x.g_index is not None
        f = piecewise_linear_integral_trapz(x.x_indices / max(x.active_step, 1), x.g_index, 0, 1)
        return f

    def f_triads(x: ScenarioStatistics) -> float:
        assert x.triads is not None
        return x.triads

    data_1 = plot_matched(ax1, metric=f_ev_count, plot_title="Event Count")
    ax1.set_title("(a) #rewiring events", loc="left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.set_yscale("log")

    data_2 = plot_matched(ax2, metric=f_ev_step, plot_title="Event Step")
    ax2.set_title("(b) norm. occurrence time", loc="left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    data_3 = plot_matched(ax3, f_auc_subj_pol_index, plot_title="AUC of Subjective Polarization Index")
    ax3.set_title(r"(c) AUC of $I_s$", loc="left")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    data_4 = plot_matched(ax4, metric=f_triads, plot_title="Triads")
    ax4.set_title("(d) #triads", loc="left")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()
    ax4.set_yscale("log")

    data_5 = plot_history(ax5, metric=f_triads, plot_title="Triads (Opinion)")
    ax5.set_title("(e) #triads, Op.", loc="left")
    ax5.grid(True, linestyle="--", alpha=0.5)
    ax5.legend()
    ax5.set_yscale("log")

    for axis, limits in ((ax1, PAPER_BAR_LIMITS.event_count), (ax2, PAPER_BAR_LIMITS.event_time),
                         (ax3, PAPER_BAR_LIMITS.subjective_auc), (ax4, PAPER_BAR_LIMITS.triads),
                         (ax5, PAPER_BAR_LIMITS.triads)):
        axis.set_ylim(*limits)
    fig.tight_layout()
    figure_dir = PATHWAY_OUTPUT.resolve() / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt_save_and_close(fig, str(figure_dir / "f_grad_index_interpret_raw"))
    write_bar_caption(figure_dir / "f_grad_index_interpret_raw", session, cutoff,
                      data_1 + data_2 + data_3 + data_4 + data_5)

    with (figure_dir / "f_grad_index_interpret_raw_data.txt").open("w") as f:
        f.write(f"Available runs: {len(session)}; descriptive I_w cutoff: {cutoff:g}; counts=low/high\n")
        for line in data_1 + data_2 + data_3 + data_4 + data_5:
            f.write(line + "\n")
