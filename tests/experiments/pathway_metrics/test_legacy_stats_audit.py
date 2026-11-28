import itertools

import numpy as np
import pytest

from ehk.metrics.homophily import normalize_homophily
from ehk.metrics.trajectory import pathway_index
from experiments.paper.pathway_metrics.legacy_stats_audit import baseline_interval, optional_mean_bounds


def test_baseline_bounds_cover_lost_clipped_values_on_a_reversing_path():
    p = np.array([.1, .9, .2, .4])
    old_h = np.array([0., .2, 0., .95])
    epsilon = .45
    old_b = epsilon-epsilon**2/8
    lower, upper = baseline_interval(p, old_h, epsilon)
    actual = []
    for zero0, zero2 in itertools.product(np.linspace(0,old_b,5), repeat=2):
        raw = old_b + (1-old_b)*old_h
        raw[[0,2]] = zero0, zero2
        h = [normalize_homophily(value,epsilon) for value in raw]
        actual.append(pathway_index(p,h))
    assert min(actual) == pytest.approx(lower)
    assert max(actual) == pytest.approx(upper)


def test_positive_archived_h_recovers_baseline_change_exactly():
    p, old_h, epsilon = [.02,.8,.06], [.1,.5,1.], .45
    old_b = epsilon-epsilon**2/8
    raw = old_b + (1-old_b)*np.array(old_h)
    actual = pathway_index(p,[normalize_homophily(value,epsilon) for value in raw])
    lower, upper = baseline_interval(p,old_h,epsilon)
    assert lower == pytest.approx(actual)
    assert upper == pytest.approx(actual)


def test_optional_mean_bounds_match_every_possible_membership_subset():
    fixed, optional = [8,12], [1,9,40]
    possible = [np.mean(fixed+[v for v,selected in zip(optional,mask) if selected])
                for mask in itertools.product((False,True),repeat=len(optional))]
    assert optional_mean_bounds(fixed,optional) == pytest.approx((min(possible),max(possible)))
