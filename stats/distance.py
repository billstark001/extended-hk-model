from typing import Dict, Union, Optional, Tuple
from numpy.typing import NDArray

import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
from scipy.integrate import quad


def ideal_dist_init(x: float):
  if x > 2:
    return 0
  return x / -2 + 1


def ideal_dist_init_array(x: NDArray):
  ret_array = x / -2 + 1
  ret_array[x > 2] = 0
  ret_array[x < 0] = 0
  return ret_array


def ideal_dist_final_o(x_max: float, delta: float = 0.01):

  h = 0.5 / delta

  def f(x: float):
    if x < delta or (x > x_max - delta and x <= x_max):
      return h
    return 0

  return f


def ideal_dist_final_s(delta: float = 0.01):

  h = 1 / delta

  def f(x: float):
    if x < delta:
      return h
    return 0

  return f


def kl_divergence(dist1_data, dist2_func, t_err=1e-13) -> float:
  dist1_func = gaussian_kde(dist1_data)

  def integrand(x):
    p_x = dist1_func(x)
    q_x = dist2_func(x)
    return p_x * np.log(p_x / q_x) if p_x > t_err and q_x > t_err else 0

  kl_divergence, _ = quad(integrand, 0, 2 + t_err)
  return kl_divergence


def kl_divergence_discrete(dist1_pmf, dist2_pmf, t_err=1e-13) -> float:

  log_dist1 = np.log(dist1_pmf)
  log_dist2 = np.log(dist2_pmf)
  log_dist1[dist1_pmf < t_err] = np.log(t_err)
  log_dist2[dist2_pmf < t_err] = np.log(t_err)

  ret_term = dist1_pmf * (log_dist1 - log_dist2)

  return np.sum(ret_term)


class DistanceCollectorContinuous:

  def __init__(
      self,
      prefix: Optional[str] = None
  ) -> None:

    self.prefix = prefix or 'distance'

  def collect(
      self,
      digraph: nx.DiGraph, graph: nx.Graph, n: int, opinion: NDArray
  ) -> Union[float, NDArray, Dict[str, Union[float, NDArray]]]:

    o_slice_mat = np.tile(opinion.reshape((opinion.size, 1)), opinion.size)
    o_sample = np.abs(o_slice_mat - o_slice_mat.T).flatten()

    neighbors = np.array(digraph.edges)
    s_sample = np.abs(
        opinion[neighbors[:, 1]] - opinion[neighbors[:, 0]])

    return {
        self.prefix + '-init-o': kl_divergence(o_sample, ideal_dist_init),
        self.prefix + '-init-s': kl_divergence(s_sample, ideal_dist_init),
        self.prefix + '-final-o': kl_divergence(o_sample, ideal_dist_final_o(o_sample.max())),
        self.prefix + '-final-s': kl_divergence(s_sample, ideal_dist_final_s()),
    }


def get_pmf(
    data: NDArray,
    hist_interval: Optional[float] = None,
    range: Optional[Tuple[float, float]] = None
):
  rmin, rmax = range if range is not None else (np.min(data), np.max(data))
  bins_orig = np.arange(rmin, rmax + hist_interval, hist_interval) \
      if hist_interval is not None else None
  count, bins = np.histogram(data, range=range) if bins_orig is None else \
      np.histogram(data, bins=bins_orig, range=range)
  axis = (bins[:-1] + bins[1:]) / 2
  dist = count / np.sum(count)
  return axis, dist


class DistanceCollectorDiscrete:

  def __init__(
      self,
      prefix: Optional[str] = None,
      hist_interval: Optional[float] = None,
      t_err: float = 1e-10
  ):
    self.prefix = prefix or 'distance'
    self.hist_interval = hist_interval
    self.t_err = t_err
    

  def collect(
      self,
      digraph: nx.DiGraph, graph: nx.Graph, n: int, opinion: NDArray
  ) -> Union[float, NDArray, Dict[str, Union[float, NDArray]]]:

    o_slice_mat = np.tile(opinion.reshape((opinion.size, 1)), opinion.size)
    o_sample = np.abs(o_slice_mat - o_slice_mat.T).flatten()

    neighbors = np.array(digraph.edges)
    s_sample = np.abs(
        opinion[neighbors[:, 1]] - opinion[neighbors[:, 0]])

    o_axis, o_pmf = get_pmf(o_sample, range=(
        0, 2), hist_interval=self.hist_interval)
    s_axis, s_pmf = get_pmf(s_sample, range=(
        0, 2), hist_interval=self.hist_interval)

    o_init = ideal_dist_init_array(o_axis)
    o_init /= np.sum(o_init)

    s_init = ideal_dist_init_array(s_axis)
    s_init /= np.sum(s_init)

    o_final = o_axis * 0
    o_final[0] = 0.5
    o_final[np.argmin(np.abs(
      o_axis - ((np.mean(opinion[opinion > 0])) - np.mean(opinion[opinion <= 0]))
    ))] = 0.5

    s_final = s_axis * 0
    s_final[0] = 1

    return {
        self.prefix + '-init-o': kl_divergence_discrete(o_pmf, o_init, t_err=self.t_err),
        self.prefix + '-init-s': kl_divergence_discrete(s_pmf, s_init, t_err=self.t_err),
        self.prefix + '-final-o': kl_divergence_discrete(o_pmf, o_final, t_err=self.t_err),
        self.prefix + '-final-s': kl_divergence_discrete(s_pmf, s_final, t_err=self.t_err),
    }
