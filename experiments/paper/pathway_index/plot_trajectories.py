import argparse
from typing import Dict, Iterable, List, Sequence, Tuple
from pathlib import Path

import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

import experiments.paper.scenarios as cfg
from common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from experiments.paper.analysis.records import ScenarioStatistics
from experiments.paper.analysis.selection import read_statistics, select_statistics, source_counts
from experiments.paper.paths import PATHWAY_OUTPUT, statistics_database

retweet_groups: List[float] = list(cfg.REPOST_AXIS)
rs_groups: List[Tuple[str, int]] = [(rs, int(retain)) for rs, retain in cfg.GRADATION_RECOMMENDATIONS.values()]

HEATMAP_BINS = 128
SUMMARY_CURVE_SAMPLES = 256
SUMMARY_CURVE_QUANTILES = 16
PAPER_INDEX_BOUNDS = (0.0, 1.0)
PAPER_AXIS_LIMITS = (-0.05, 1.05)
PAPER_INDEX_TICKS = tuple(np.linspace(0.0, 1.0, 6))


def _safe_curve(p_index: np.ndarray | None, h_index: np.ndarray | None) -> np.ndarray | None:
    if p_index is None or h_index is None:
        return None
    if p_index.ndim != 1 or h_index.ndim != 1 or p_index.shape != h_index.shape:
        raise ValueError("trajectory indices must be equal-length one-dimensional arrays")
    if p_index.size < 2 or h_index.size < 2:
        return None
    mask = np.isfinite(p_index) & np.isfinite(h_index)
    if np.count_nonzero(mask) < 2:
        return None

    points = np.column_stack((p_index[mask], h_index[mask]))
    if points.shape[0] < 2:
        return None
    return points


def _group_data(
    full_data: Iterable[ScenarioStatistics],
) -> Dict[Tuple[float, str, int], List[Tuple[np.ndarray, float]]]:
    grouped: Dict[Tuple[float, str, int], List[Tuple[np.ndarray, float]]] = {}
    for rt in retweet_groups:
        for rs_type, retain_count in rs_groups:
            grouped[(rt, rs_type, retain_count)] = []

    for datum in full_data:
        if datum.pathway_integral is None:
            continue
        if datum.p_index is None or datum.h_index is None:
            continue

        retain_count = int(round(float(datum.post_retain_count)))
        key = (float(datum.repost_rate), str(datum.recsys_type), retain_count)
        if key not in grouped:
            continue

        points = _safe_curve(datum.p_index, datum.h_index)
        if points is None:
            continue

        grad = float(datum.pathway_integral)
        grouped[key].append((points, grad))

    return grouped


def _density_image(
    curves: List[Tuple[np.ndarray, float]], cmap, norm: Normalize, p_bounds: tuple[float, float], bins: int = HEATMAP_BINS
) -> np.ndarray | None:
    if not curves:
        return None

    counts = np.zeros((bins, bins), dtype=float)
    grad_sums = np.zeros((bins, bins), dtype=float)

    for points, grad in curves:
        hist, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=(p_bounds, (0.0, 1.0)))
        weighted_hist, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=bins,
            range=(p_bounds, (0.0, 1.0)),
            weights=np.full(points.shape[0], grad, dtype=float),
        )
        counts += hist
        grad_sums += weighted_hist

    if counts.max() <= 0:
        return None

    mean_grad = np.full_like(counts, 0.5)
    valid = counts > 0
    mean_grad[valid] = grad_sums[valid] / counts[valid]

    density = np.log1p(counts)
    density /= density.max()
    density = density.T
    mean_grad = mean_grad.T

    rgba = cmap(norm(mean_grad))
    rgba[..., :3] = 1.0 - (1.0 - rgba[..., :3]) * density[..., None]
    rgba[..., 3] = np.where(density > 0, 0.95, 0.0)
    return rgba


def resample_curve(points: np.ndarray, n_samples: int = SUMMARY_CURVE_SAMPLES) -> np.ndarray:
    if points.shape[0] == n_samples:
        return points

    src = np.linspace(0.0, 1.0, points.shape[0])
    dst = np.linspace(0.0, 1.0, n_samples)
    return np.column_stack((np.interp(dst, src, points[:, 0]), np.interp(dst, src, points[:, 1])))


def summary_curves_clustered(
    curves: List[Tuple[np.ndarray, float]], n_groups: int = SUMMARY_CURVE_QUANTILES
) -> List[Tuple[np.ndarray, float, int]]:
    if not curves:
        return []

    # Resample every curve to the common coordinate system.
    resampled_curves = [resample_curve(points) for points, _ in curves]
    resampled_array = np.stack(resampled_curves, axis=0)

    n_curves, n_samples, n_dims = resampled_array.shape

    # Flatten coordinates so K-means uses pointwise Euclidean distance.
    features = resampled_array.reshape(n_curves, n_samples * n_dims)

    # Do not request more spatial trajectory clusters than samples.
    n_clusters = min(n_groups, n_curves)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Aggregate each trajectory cluster.
    grads = [grad for _, grad in curves]
    summaries: List[Tuple[np.ndarray, float, int]] = []

    for cluster_idx in range(n_clusters):
        # Select curves in this cluster.
        indices = np.where(labels == cluster_idx)[0]
        if len(indices) == 0:
            continue

        # Collect their resampled coordinates.
        cluster_curves = resampled_array[indices]

        # Use the median curve as the cluster backbone.
        median_curve = np.median(cluster_curves, axis=0)

        # Retain the mean pathway index for scientific color encoding.
        mean_grad = float(np.mean([grads[i] for i in indices]))

        summaries.append((median_curve, mean_grad, len(indices)))

    # Sort by pathway index for stable layering and legends.
    summaries.sort(key=lambda x: x[1])

    return summaries


def plot_p_h_index_trajectories(session: Sequence[ScenarioStatistics]) -> Figure:
    setup_paper_params()

    full_data: Iterable[ScenarioStatistics] = select_statistics(
        session,
        lambda row: row.name.startswith("s_grad"),
        lambda row: row.p_index is not None,
        lambda row: row.h_index is not None,
        lambda row: row.pathway_integral is not None,
    )

    grouped = _group_data(full_data)

    fig, axes = plt_figure(n_row=4, n_col=4, hw_ratio=1, total_width=13.5, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.08, top=0.96, wspace=0.10, hspace=0.14)

    axes_arr = np.array(axes)
    cmap = colormaps["coolwarm"]
    norm = Normalize(vmin=0.0, vmax=1.0)

    for i_rt, rt in enumerate(retweet_groups):
        for i_rs, (rs_type, retain_count) in enumerate(rs_groups):
            ax = axes_arr[i_rt, i_rs]
            curves = grouped[(rt, rs_type, retain_count)]

            density_img = _density_image(curves, cmap, norm, PAPER_INDEX_BOUNDS)
            if density_img is not None:
                ax.imshow(
                    density_img,
                    extent=(*PAPER_INDEX_BOUNDS, 0, 1),
                    origin="lower",
                    interpolation="bilinear",
                    aspect="auto",
                    zorder=1,
                )

            for points, grad, _ in summary_curves_clustered(curves):
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    color=cmap(norm(grad)),
                    linewidth=1.1,
                    alpha=0.95,
                    solid_capstyle="round",
                    zorder=2,
                )

            ax.set_xlim(*PAPER_AXIS_LIMITS)
            ax.set_ylim(*PAPER_AXIS_LIMITS)
            ax.set_xticks(PAPER_INDEX_TICKS)
            ax.set_yticks(PAPER_INDEX_TICKS)
            ax.set_facecolor("white")
            ax.grid(True, linestyle="--", linewidth=0.5)

            title_rs = "St" if rs_type == "StructureM9" else "Op"
            ax.set_title(f"p={rt:g}, {title_rs}/k={retain_count}", loc="left", fontsize=9)

            if i_rt == 3:
                ax.set_xlabel(r"$I_p$")
            else:
                ax.set_xticklabels([])

            if i_rs == 0:
                ax.set_ylabel(r"$I_h$")
            else:
                ax.set_yticklabels([])

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes((0.91, 0.17, 0.015, 0.66))
    cbar = fig.colorbar(sm, cax=cax, ticks=PAPER_INDEX_TICKS)
    cbar.set_label(r"mean $I_w$")

    return fig


def write_trajectory_caption(stem: Path, session: Sequence[ScenarioStatistics]) -> None:
    selected = select_statistics(session, lambda row: row.name.startswith("s_grad"))
    groups = _group_data(selected)
    lines = [
        f"Available completed trajectories: n={len(session)}.",
        "Panels pool rewiring/influence settings; each n counts valid trajectories in that repost/policy/history group.",
        "Density uses saved adaptive samples; cluster backbones are medians after resampling to a shared point-index grid.",
        "Color represents mean I_w. Empty groups contain no trajectory.",
        "The paper's fixed coolwarm scale is [0,1], with both index axes shown on [-0.05,1.05].",
        "Saved trajectories and I_w values remain unchanged; values outside the color scale use endpoint colors, and portions outside the axes are not shown.",
    ]
    if session:
        lines.append(f"Origins (available runs): {source_counts(session)}; "
                     f"analysis_id: {', '.join(sorted({row.analysis_id for row in session}))}.")
    lines.extend(f"p={repost:g}, {policy}/k={history}: n={len(curves)}"
                 for (repost, policy, history), curves in groups.items())
    stem.with_name(stem.name + "_caption.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the original policy/history trajectory facets.")
    parser.add_argument("--database", type=Path, default=statistics_database())
    parser.add_argument("--origin", action="append", help="Repeat to explicitly pool named sources.")
    parser.add_argument("--analysis-id")
    parser.add_argument("--output", type=Path, default=PATHWAY_OUTPUT.resolve())
    args = parser.parse_args(argv)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    session = read_statistics(args.database, mode="gradation", origins=args.origin, analysis_id=args.analysis_id)
    fig = plot_p_h_index_trajectories(session)
    plt_save_and_close(fig, str(figure_dir / "f_p_h_index_trajectory"), jpg=True)
    write_trajectory_caption(figure_dir / "f_p_h_index_trajectory", session)


if __name__ == "__main__":
    main()
