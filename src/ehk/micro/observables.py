from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

from common.numerics import adaptive_discrete_sampling, area_under_curve, first_more_or_equal_than
from common.pipeline import Context
from ehk.metrics.evaluation import MeasurementEvaluator
from ehk.metrics.homophily import mean_followee_edge_concordance, normalize_homophily
from ehk.metrics.pathway import OBSERVATION_INTERVAL, first_passage
from ehk.metrics.polarization import DistanceCalculator, opinion_peak_count
from ehk.metrics.trajectory import pathway_index

if TYPE_CHECKING:
    from smp_bindings import PostEventBody, RawSimulationRecord


c = Context(ignore_prefix="get_")
c.set_state(active_threshold=0.98, min_inactive_value=0.75, opinion_peak_distance=50, endpoint_complement=True)


def floyd_warshall_weight_matrix(A: NDArray):
    n, m = A.shape
    np.fill_diagonal(A, 0)  # diagonal elements should be zero
    for i in range(n):
        # The second term has the same shape as A due to broadcasting
        A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
    return A


def get_fw_stats(A_in: NDArray):
    # A_in: adjacency boolean array (nonedge = 0)
    A = np.copy(A_in).astype(float)
    A[A > 1] = 1
    A[A == 0] = np.inf
    A_graph = np.minimum(A, A.T)
    A_fw = floyd_warshall_weight_matrix(A_graph)
    return A_fw


def get_triads_stats(A: NDArray):
    # A: adjacency boolean array (nonedge = 0)
    A2 = A @ A
    A_triads = np.copy(A2)
    A_triads[A == 0] = 0
    n_triads = np.sum(A_triads)

    return n_triads, A_triads


def get_opinion_diff_vars(adj_mat: NDArray, opinion: NDArray):
    out_deg = np.sum(adj_mat, axis=1).reshape(-1, 1)
    o = opinion.reshape(-1, 1)
    o_adj_sum = adj_mat @ o
    o_cur_sum = out_deg * o
    o2_adj_sum = adj_mat @ (o**2)
    o2_cur_sum = out_deg * (o**2)

    var_term_1_sum = o2_adj_sum + o2_cur_sum - 2 * o_adj_sum * o
    var_term_2_sum = o_adj_sum - o_cur_sum

    var = var_term_1_sum / out_deg - (var_term_2_sum / out_deg) ** 2

    return var.flatten()


def bimodality_coefficient(data: NDArray):
    """
    Calculate the Bimodality Coefficient (BC) for a given dataset.

    Parameters:
    data (numpy.array): Input data

    Returns:
    float: Bimodality Coefficient
    """
    n = data.size
    m3 = skew(data, bias=False)
    m4 = kurtosis(data, bias=False)
    numerator = m3**2 + 1
    denominator = m4 + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))

    bc = float(numerator / denominator)
    return bc


def get_bc_hom(G: nx.DiGraph, opinion: NDArray):
    edges = np.array(G.edges())
    paired_opinion = opinion[edges]
    rotated_opinion = np.sum(paired_opinion, axis=1) / 2
    return bimodality_coefficient(rotated_opinion)


@c.selector
def get_total_steps(scenario_record: RawSimulationRecord):
    return scenario_record.max_step


@c.selector
def get_rewiring_event_stats(scenario_record: RawSimulationRecord):
    from smp_bindings import get_events_by_step_range

    rewiring_events = get_events_by_step_range(scenario_record.events_db, 0, scenario_record.max_step + 1, "Rewiring")
    event_step = np.array([x.step for x in rewiring_events])
    return event_step


@c.selector
def get_retweeted_tweets_lifecycle_raw_stats(scenario_record: RawSimulationRecord):
    from smp_bindings import batch_load_event_bodies, get_events_by_step_range

    reposted_posts = get_events_by_step_range(scenario_record.events_db, 0, scenario_record.max_step + 1, "Post")
    batch_load_event_bodies(scenario_record.events_db, reposted_posts)
    tweet_occurrence_step = defaultdict(int)
    tweet_opinions: Dict[Tuple[int, int], float] = {}

    for tweet in reposted_posts:
        tweet_body: PostEventBody | None = tweet.body
        assert tweet_body is not None and tweet_body.is_repost, "Corrupted data!"
        tweet_pair = tweet_body.record.agent_id, tweet_body.record.step
        tweet_occurrence_step[tweet_pair] = max(tweet_occurrence_step[tweet_pair], tweet.step)
        tweet_opinions[tweet_pair] = tweet_body.record.opinion

    lifespans = []
    opinions = []
    for tweet_pair, last_step in tweet_occurrence_step.items():
        init_step = tweet_pair[1]
        lifespans.append([init_step, last_step])
        opinions.append(tweet_opinions[tweet_pair])

    retweeted_lifespans = np.array(lifespans, dtype=int)
    retweeted_opinions = np.array(opinions, dtype=float)

    return retweeted_lifespans, retweeted_opinions


distance_collector = DistanceCalculator(use_js_divergence=True, min_bandwidth=0.01, sample_count=500)


def calc_distance(rec: RawSimulationRecord, step: int):
    """Evaluate the adopted B=4081 distance indices on one saved state."""
    return MeasurementEvaluator(calculator=distance_collector).distance(rec, int(step))


def calc_distance_components(rec: RawSimulationRecord, step: int):
    """Retain original exact KDE and archived directions for explicit audits."""
    result = distance_collector.calculate(
        rec.get_graph(step), rec.opinions[step], rec.metadata["HKParams"]["Tolerance"], random_state=step
    )
    return result.rand_o, result.rand_s, 1 - result.worst_o, 1 - result.worst_s


def calc_homophily(rec: RawSimulationRecord, step: int):
    epsilon = rec.metadata["HKParams"]["Tolerance"]
    rho = mean_followee_edge_concordance(rec.get_graph(step), rec.opinions[step], epsilon)
    return normalize_homophily(rho, epsilon) if np.isfinite(rho) else np.nan


def followee_post_concordance(rec: RawSimulationRecord, step: int) -> float:
    # Row t describes the feed used to update the model from t-1 to t.
    counts = rec.feed_counts(step)
    if counts is None:
        return np.nan
    shown = counts[:, 0].astype(float) + counts[:, 2]
    seen = shown > 0
    return float(np.mean(counts[seen, 0] / shown[seen])) if np.any(seen) else np.nan


@c.selector(("x_indices", "h_index", "p_index", "g_index"))
def get_indices(scenario_record: RawSimulationRecord):
    return MeasurementEvaluator(calculator=distance_collector).indices(scenario_record)


@c.selector
def get_concordance_series(scenario_record: RawSimulationRecord, x_indices: NDArray):
    epsilon = scenario_record.metadata["HKParams"]["Tolerance"]
    rho_edge = np.array(
        [
            mean_followee_edge_concordance(scenario_record.get_graph(int(t)), scenario_record.opinions[t], epsilon)
            for t in x_indices
        ]
    )
    rho_post = np.array([followee_post_concordance(scenario_record, int(t)) for t in x_indices])
    return rho_edge, rho_post


@c.selector(("pathway_label", "separation_step", "segregation_step"))
def get_pathway_passage(scenario_record: RawSimulationRecord, probe_binary: Path):
    end = scenario_record.max_step
    epsilon = scenario_record.metadata["HKParams"]["Tolerance"]
    steps = [*range(0, end, OBSERVATION_INTERVAL), end]
    measured = scenario_record.evaluate_probe(
        steps, measurements=["pathway"], epsilon=epsilon, binary_path=str(probe_binary),
    )
    if measured["source_max_step"] != end or [state["step"] for state in measured["results"]] != steps:
        raise ValueError("native pathway observations differ from the requested saved history")
    passage = first_passage(
        (state["step"], state["pathway"]["separation"], state["pathway"]["excess_concordance"])
        for state in measured["results"]
    )
    return passage.label, passage.separation_step, passage.segregation_step


@c.selector
def get_backdrop_rates(p_index: NDArray, h_index: NDArray, g_index: NDArray):
    p_index_abs = np.abs(p_index[1:] - p_index[:-1])
    h_index_abs = np.abs(h_index[1:] - h_index[:-1])
    g_index_abs = np.abs(g_index[1:] - g_index[:-1])

    p_backdrop = np.sum(p_index_abs)
    h_backdrop = np.sum(h_index_abs)
    g_backdrop = np.sum(g_index_abs)

    return p_backdrop, h_backdrop, g_backdrop


def get_gradation_index_hp(p_index, h_index, *, endpoint_complement=True):
    """Compute I_w, including the terminal contribution unless explicitly disabled."""
    return pathway_index(p_index, h_index, endpoint_complement=endpoint_complement)


@c.selector("gradation_index_hp")
def _selected_gradation_index_hp(p_index, h_index, endpoint_complement):
    return get_gradation_index_hp(p_index, h_index, endpoint_complement=endpoint_complement)


@c.selector("metric_audit")
def get_metric_audit(
    scenario_record: RawSimulationRecord, x_indices, p_index, g_index, h_index, rho_post, endpoint_complement=True
):
    """Recompute historical conventions from raw states, without replacing the main indices.

    A missing initial feed is never imputed. All staged convention comparisons
    use the same available observations and disclose their actual prefix.
    """
    legacy = np.array([calc_distance_components(scenario_record, int(t))[2:] for t in x_indices])
    epsilon = scenario_record.metadata["HKParams"]["Tolerance"]
    old_baseline = epsilon - epsilon**2 / 8
    new_baseline = epsilon - epsilon**2 / 4
    legacy_h = np.maximum(0, (rho_post - old_baseline) / (1 - old_baseline))
    corrected_post_h = np.maximum(0, (rho_post - new_baseline) / (1 - new_baseline))
    valid = np.isfinite(legacy[:, 0]) & np.isfinite(p_index) & np.isfinite(legacy_h) & np.isfinite(h_index)

    def integral(p, h, *, complement=True):
        if not np.any(valid):
            return float("nan")
        return area_under_curve(np.array([p[valid], h[valid]]), complement=1 if complement else None)

    variants = {
        "legacy_reconstructed": integral(legacy[:, 0], legacy_h),
        "correct_uniform_baseline": integral(legacy[:, 0], corrected_post_h),
        "use_edge_endpoints": integral(legacy[:, 0], h_index),
        "use_random_reference_distance": integral(np.sqrt(np.maximum(0, p_index)), h_index),
        "use_js_divergence": integral(p_index, h_index),
        "remove_endpoint_complement": integral(p_index, h_index, complement=False),
        "revised_full_trajectory": get_gradation_index_hp(p_index, h_index, endpoint_complement=endpoint_complement),
        "full_trajectory_with_complement": get_gradation_index_hp(p_index, h_index, endpoint_complement=True),
        "full_trajectory_without_complement": get_gradation_index_hp(p_index, h_index, endpoint_complement=False),
    }
    return {
        "legacy_p_index": legacy[:, 0],
        "legacy_g_index": legacy[:, 1],
        "legacy_h_index": legacy_h,
        "corrected_post_h_index": corrected_post_h,
        "legacy_pathway_integral": variants["legacy_reconstructed"],
        "pathway_variants": variants,
        "audit_prefix_start": int(x_indices[valid][0]) if np.any(valid) else None,
        "audit_missing_steps": np.asarray(x_indices)[~valid],
        "audit_sampling": "revised adaptive observation grid; convention comparisons share finite observations",
        "audit_endpoint_complement": endpoint_complement,
    }


def interp(x0, x1, p):
    return x0 * (1 - p) + x1 * p


@c.selector
def calc_active_step(x_indices: np.ndarray, g_index: np.ndarray, active_threshold: float, min_inactive_value: float):

    if x_indices.size == 0 or g_index.size == 0:
        return 0, 0, 0.0, np.zeros(0, dtype=float), 0.0

    finite_mask = np.isfinite(x_indices) & np.isfinite(g_index)
    if not np.any(finite_mask):
        return 0, 0, 0.0, np.zeros(0, dtype=float), 0.0
    if not np.all(finite_mask):
        x_indices = x_indices[finite_mask]
        g_index = g_index[finite_mask]

    if x_indices.size == 1:
        single_step = int(x_indices[0])
        single_value = float(g_index[0])
        return 0, single_step, single_value, g_index.copy(), single_value

    active_step_crit_value = max(float(g_index[-1]) * active_threshold, min_inactive_value)
    if not np.any(g_index >= active_step_crit_value):
        return len(x_indices) - 1, int(x_indices[-1]), float(g_index[-1]), g_index.copy(), float(np.mean(g_index))
    if g_index[0] >= active_step_crit_value:
        return 0, int(x_indices[0]), float(g_index[0]), g_index[:1].copy(), float(g_index[0])
    active_step_index = int(first_more_or_equal_than(np.array(g_index), active_step_crit_value))
    if active_step_index >= len(x_indices):
        active_step_index = len(x_indices) - 1  # take last index
    elif active_step_index < 1:
        active_step_index = 1

    post_active_step = x_indices[active_step_index]
    post_active_step_g = g_index[active_step_index]
    pre_active_step = x_indices[active_step_index - 1]
    pre_active_step_g = g_index[active_step_index - 1]

    denom = post_active_step_g - pre_active_step_g
    if abs(denom) < 1e-12:
        interp_p = 0.5
    else:
        interp_p = max(min((active_step_crit_value - pre_active_step_g) / denom, 1), 0)
    active_step = int(interp(pre_active_step, post_active_step, interp_p))
    active_step_threshold = float(interp(pre_active_step_g, post_active_step_g, interp_p))

    g_index_active = g_index[:active_step_index]
    g_index_mean_active = np.mean(g_index_active)

    return active_step_index, active_step, active_step_threshold, g_index_active, g_index_mean_active


@c.selector
def calc_opinion_last(scenario_record: RawSimulationRecord):
    opinion_last = scenario_record.opinions[scenario_record.max_step]
    opinion_last_mean = float(np.mean(opinion_last))
    opinion_last_diff = float(
        np.mean(opinion_last[opinion_last > opinion_last_mean])
        - np.mean(opinion_last[opinion_last <= opinion_last_mean])
    )
    return opinion_last, opinion_last_mean, opinion_last_diff


@c.selector
def get_opinion_diff(scenario_record: RawSimulationRecord):
    opinion = scenario_record.opinions
    opinion_diff = np.copy(opinion)
    opinion_diff[1:] -= opinion[:-1]
    opinion_diff[0] = 0
    return opinion_diff


@c.selector
def get_opinion_diff_mean(opinion_diff: NDArray, scenario_record: RawSimulationRecord):
    odm_raw = np.abs(opinion_diff).mean(axis=1)
    x_opinion_diff_mean, opinion_diff_mean_smpl = adaptive_discrete_sampling(
        lambda x: float(odm_raw[x]), 0.01, 0, scenario_record.max_step, max_interval=512
    )
    return x_opinion_diff_mean, opinion_diff_mean_smpl


@c.selector
def get_opinion_decrease_speed(opinion_diff: NDArray, active_step: float):
    opinions_diff_abs = np.abs(opinion_diff)

    step_mask = np.arange(opinions_diff_abs.shape[0], dtype=int)

    points = []
    seg = 0.2
    x_seg = np.arange(0, 1, seg)
    for seg_start in x_seg:
        seg_end = seg_start + seg
        step_mask_seg = np.logical_and(step_mask >= seg_start * active_step, step_mask < seg_end * active_step)
        sum_seg = np.sum(opinions_diff_abs[step_mask_seg], axis=0)
        mean_seg = np.mean(sum_seg)
        std_seg = np.std(sum_seg)
        points.append((mean_seg, std_seg))

    opinion_diff_seg_mean, opinion_diff_seg_std = np.array(points).T

    return opinion_diff_seg_mean, opinion_diff_seg_std


@c.selector
def get_n_triads_and_bc_hom(scenario_record: RawSimulationRecord):
    last_graph = scenario_record.get_graph(scenario_record.max_step)
    last_opinion = scenario_record.opinions[scenario_record.max_step]
    adj_mat = nx.to_numpy_array(last_graph, multigraph_weight=min)
    n_triads, _ = get_triads_stats(adj_mat)
    n_triads = int(n_triads)

    bc_hom_last = get_bc_hom(last_graph, last_opinion)

    return n_triads, bc_hom_last


@c.selector
def get_last_community_count(scenario_record: RawSimulationRecord):
    import json
    from collections import Counter

    import igraph as ig
    import leidenalg

    last_graph = scenario_record.get_graph(scenario_record.max_step)
    edges = list(last_graph.edges())
    igraph_g = ig.Graph(directed=True)
    igraph_g.add_vertices(list(last_graph.nodes()))
    igraph_g.add_edges(edges)

    partition = leidenalg.find_partition(igraph_g, leidenalg.ModularityVertexPartition)

    membership = partition.membership
    result = dict(zip(last_graph.nodes(), membership))

    last_community_sizes_dict = dict(Counter(result.values()))
    last_community_count = len(last_community_sizes_dict)

    last_community_sizes = json.dumps(last_community_sizes_dict)

    return last_community_count, last_community_sizes


@c.selector("last_opinion_peak_count")
def get_last_opinion_peak_count(opinion_last: NDArray, opinion_peak_distance: int):

    return opinion_peak_count(opinion_last, opinion_peak_distance)


@c.selector
def get_mean_vars_x_and_smpl(scenario_record: RawSimulationRecord):
    def get_vars(step: float):
        g = scenario_record.get_graph(int(step))
        o = scenario_record.opinions[int(step)]
        A = nx.to_numpy_array(g, multigraph_weight=min)  # adjacency matrix
        vars = get_opinion_diff_vars(A, o)
        return np.mean(vars)

    x_mean_vars, mean_vars_smpl = adaptive_discrete_sampling(
        get_vars, 0.01, 0, scenario_record.max_step, max_interval=512, int_midpoint=True
    )

    return x_mean_vars, mean_vars_smpl


@c.selector
def get_bc_hom_x_and_smpl(scenario_record: RawSimulationRecord):
    def get_bc_hom_step(step: float):
        g = scenario_record.get_graph(int(step))
        o = scenario_record.opinions[int(step)]
        ret = get_bc_hom(g, o)
        if np.isnan(ret):
            return None
        return ret

    def err_bc_hom(x1, x2, x3, t3):
        if x1 is None and x2 is None and x3 is None:
            return 0
        if x1 is None or x2 is None or x3 is None:
            return 1
        ret = x1 * (1 - t3) + x2 * t3 - x3
        return ret

    x_bc_hom, bc_hom_smpl = adaptive_discrete_sampling(
        get_bc_hom_step, 0.01, 0, scenario_record.max_step, max_interval=512, err_func=err_bc_hom, int_midpoint=True
    )

    return x_bc_hom, bc_hom_smpl


_, state_vars, cycles = c.get_dep_graph()

valid_state_vars = {
    "scenario_record",
    "active_threshold",
    "min_inactive_value",
    "opinion_peak_distance",
    "endpoint_complement",
    "probe_binary",
}

invalid_state_vars = [x for x in state_vars if x not in valid_state_vars]

if invalid_state_vars:
    raise ValueError("Invalid state variables detected: " + ",".join([str(c) for c in invalid_state_vars]))

if cycles:
    raise ValueError("Cyclic dependencies detected: " + ",".join([str(c) for c in cycles]))
