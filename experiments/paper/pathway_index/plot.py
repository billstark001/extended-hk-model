from functools import partial

import numpy as np

from common.plotting import PAPER_WIDTH, plt_figure, plt_save_and_close, setup_paper_params
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import descriptive_iw_cutoff, read_statistics
from experiments.paper.paths import PATHWAY_OUTPUT, statistics_database
from experiments.paper.pathway_index.bar_plot import PAPER_BAR_LIMITS, plot_grouped_bars, write_bar_caption
from experiments.paper.pathway_index.data_utils import piecewise_linear_integral_trapz

setup_paper_params()


# First row filters: distinguish by recommendation strategy (St vs Op), ignore retweet status
k_filters_recsys = [(lambda row: row.recsys_type == "StructureM9",), (lambda row: row.recsys_type == "OpinionM9",)]
k_filters_recsys = [(*filters, lambda row: row.post_retain_count == 0) for filters in k_filters_recsys]

k_labels_recsys = ["St.", "Op."]

# Second row filters: distinguish by retweet status (!R vs R), ignore recommendation strategy
k_filters_retweet = [(lambda row: row.repost_rate == 0,), (lambda row: row.repost_rate != 0,)]
k_filters_retweet = [(*filters, lambda row: row.post_retain_count == 0) for filters in k_filters_retweet]

k_labels_retweet = ["!R", "R"]


if __name__ == "__main__":
    session = read_statistics(statistics_database(), mode="gradation")

    cutoff = descriptive_iw_cutoff()
    plot_recsys = partial(plot_grouped_bars, records=session, cutoff=cutoff,
                         category_filters=k_filters_recsys, category_labels=k_labels_recsys)
    plot_retweet = partial(plot_grouped_bars, records=session, cutoff=cutoff,
                          category_filters=k_filters_retweet, category_labels=k_labels_retweet)
    fig, axes = plt_figure(n_row=2, n_col=4, total_width=PAPER_WIDTH * 0.8, hw_ratio=4 / 3,
                           constrained_layout=False)

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

    # First row (a-d): comparing recommendation strategies (St vs Op)
    data_a = plot_recsys(axes[0][0], metric=f_ev_count, plot_title="Event Count by RecSys")
    axes[0][0].set_title("(a)", loc="left")
    axes[0][0].set_ylabel("#rewiring events")
    axes[0][0].grid(True, linestyle="--", alpha=0.5)
    axes[0][0].set_yscale("log")

    data_b = plot_recsys(axes[0][1], metric=f_ev_step, plot_title="Event Step by RecSys")
    axes[0][1].set_title("(b)", loc="left")
    axes[0][1].set_ylabel("norm. occurrence time")
    axes[0][1].grid(True, linestyle="--", alpha=0.5)

    data_c = plot_recsys(
        axes[0][2],
        metric=f_auc_subj_pol_index,
        plot_title="AUC of Subjective Polarization Index by RecSys",
    )
    axes[0][2].set_title("(c)", loc="left")
    axes[0][2].set_ylabel(r"AUC of $I_s$")
    axes[0][2].grid(True, linestyle="--", alpha=0.5)

    data_d = plot_recsys(axes[0][3], metric=f_triads, plot_title="Triads by RecSys")
    axes[0][3].set_title("(d)", loc="left")
    axes[0][3].set_ylabel("#triads")
    axes[0][3].grid(True, linestyle="--", alpha=0.5)
    axes[0][3].legend(fontsize=8)
    axes[0][3].set_yscale("log")

    # Second row (e-h): comparing retweet status (!R vs R)
    data_e = plot_retweet(axes[1][0], metric=f_ev_count, plot_title="Event Count by Retweet")
    axes[1][0].set_title("(e)", loc="left")
    axes[1][0].set_ylabel("#rewiring events")
    axes[1][0].grid(True, linestyle="--", alpha=0.5)
    axes[1][0].set_yscale("log")

    data_f = plot_retweet(axes[1][1], metric=f_ev_step, plot_title="Event Step by Retweet")
    axes[1][1].set_title("(f)", loc="left")
    axes[1][1].set_ylabel("norm. occurrence time")
    axes[1][1].grid(True, linestyle="--", alpha=0.5)

    data_g = plot_retweet(
        axes[1][2],
        metric=f_auc_subj_pol_index,
        plot_title="AUC of Subjective Polarization Index by Retweet",
    )
    axes[1][2].set_title("(g)", loc="left")
    axes[1][2].set_ylabel(r"AUC of $I_s$")
    axes[1][2].grid(True, linestyle="--", alpha=0.5)

    data_h = plot_retweet(axes[1][3], metric=f_triads, plot_title="Triads by Retweet")
    axes[1][3].set_title("(h)", loc="left")
    axes[1][3].set_ylabel("#triads")
    axes[1][3].grid(True, linestyle="--", alpha=0.5)
    axes[1][3].legend(fontsize=8)
    axes[1][3].set_yscale("log")

    for row in axes:
        for axis, limits in zip(row, (PAPER_BAR_LIMITS.event_count, PAPER_BAR_LIMITS.event_time,
                                     PAPER_BAR_LIMITS.subjective_auc, PAPER_BAR_LIMITS.triads)):
            axis.set_ylim(*limits)
    fig.tight_layout()
    figure_dir = PATHWAY_OUTPUT.resolve() / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt_save_and_close(fig, str(figure_dir / "f_grad_index_interpret"))
    write_bar_caption(figure_dir / "f_grad_index_interpret", session, cutoff,
                      data_a + data_b + data_c + data_d + data_e + data_f + data_g + data_h)

    with (figure_dir / "f_grad_index_interpret_data.txt").open("w") as f:
        f.write(f"Available runs: {len(session)}; descriptive I_w cutoff: {cutoff:g}; counts=low/high\n")
        for line in data_a + data_b + data_c + data_d + data_e + data_f + data_g + data_h:
            f.write(line + "\n")
