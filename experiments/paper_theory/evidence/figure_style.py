"""Main-study colors, labels and epsilon-specific table/figure selection."""

from matplotlib.lines import Line2D

from common.io.tables import numeric_rows
from common.plotting.figures import save as save_figure

from .protocol import KERNELS

COLORS = ("#555555", "#86b5d8", "#166da8", "#eba981", "#bd4e19")
LABELS = ("Random", "Opinion1", "Opinion4", "Structure1", "Structure4")
RULES = ("hk", "deffuant")


def read_table(directory, name, epsilon=None):
    rows = numeric_rows(directory / "tables" / (name + ".csv"))
    return rows if epsilon is None else [r for r in rows if r.get("epsilon") == epsilon]


def matches(row, **values):
    return all(row[k] == v for k, v in values.items())


def save(figure, directory, name, epsilon=None):
    if epsilon is not None:
        title = figure._suptitle.get_text() if figure._suptitle is not None else ""
        figure.suptitle(f"epsilon={epsilon:g}" + ("; " + title if title else ""), fontsize=11)
    save_figure(figure, directory / name)


def clock_label(horizon):
    return rf"Endpoint-referenced residual area ($\tau_{{max}}={horizon:g}$)"


def color_legend(kernels=None):
    return [
        Line2D([], [], color=c, label=label)
        for c, label, kernel in zip(COLORS, LABELS, KERNELS)
        if kernels is None or kernel in kernels
    ]
