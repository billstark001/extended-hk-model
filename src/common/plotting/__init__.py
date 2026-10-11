"""Public figure sizing, publication style and plotting helpers."""

from .core import (
    DEFAULT_HW_RATIO,
    PAPER_WIDTH,
    draw_adaptive_moving_stats,
    get_colormap,
    linear_func,
    numpy_to_latex_table,
    plot_line,
    plt_figure,
    plt_save_and_close,
    setup_paper_params,
)

__all__ = [
    "PAPER_WIDTH",
    "DEFAULT_HW_RATIO",
    "setup_paper_params",
    "plt_figure",
    "plt_save_and_close",
    "get_colormap",
    "linear_func",
    "plot_line",
    "draw_adaptive_moving_stats",
    "numpy_to_latex_table",
]
