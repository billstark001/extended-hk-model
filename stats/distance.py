from typing import Dict, Union, Optional, Tuple
from numpy.typing import NDArray

import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
from scipy.integrate import quad


def ideal_dist_init_array(x: NDArray):
  ret_array = x / -2 + 1
  ret_array[x > 2] = 0
  ret_array[x < 0] = 0
  return ret_array


def kl_divergence_discrete(dist1_pmf, dist2_pmf, t_err=1e-13) -> float:

  log_dist1 = np.log(dist1_pmf + t_err)
  log_dist2 = np.log(dist2_pmf + t_err)
  
  ret_term = dist1_pmf * (log_dist1 - log_dist2)

  return np.sum(ret_term)

def js_divergence_discrete(dist1_pmf, dist2_pmf, t_err = 1e-13) -> float:
  
  log_dist1 = np.log(dist1_pmf + t_err)
  log_dist2 = np.log(dist2_pmf + t_err)
  log_dist3 = np.log((dist1_pmf + dist2_pmf) / 2 + t_err)
  
  ret_nom = dist1_pmf * (log_dist1 - log_dist3) + dist2_pmf * (log_dist2 - log_dist3)
  return (np.sum(ret_nom) / 2) ** 0.5
  


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
      t_err: float = 1e-10,
      use_js_divergence: bool = False,
  ):
    self.prefix = prefix or 'distance'
    self.hist_interval = hist_interval
    self.t_err = t_err
    self.use_js_divergence = use_js_divergence
    self.div = js_divergence_discrete if use_js_divergence else kl_divergence_discrete

  def collect(
      self,
      digraph: nx.DiGraph, graph: nx.Graph, n: int, opinion: NDArray
  ) -> Union[float, NDArray, Dict[str, Union[float, NDArray]]]:

    # sampling

    o_slice_mat = np.tile(opinion.reshape((opinion.size, 1)), opinion.size)
    o_sample = np.abs(o_slice_mat - o_slice_mat.T).flatten()

    neighbors = np.array(digraph.edges)
    s_sample = np.abs(
        opinion[neighbors[:, 1]] - opinion[neighbors[:, 0]])

    # calculate pmf

    o_axis, o_pmf = get_pmf(o_sample, range=(
        0, 2), hist_interval=self.hist_interval)
    s_axis, s_pmf = get_pmf(s_sample, range=(
        0, 2), hist_interval=self.hist_interval)

    # best and worst cases

    o_best = ideal_dist_init_array(o_axis)
    o_best /= np.sum(o_best)

    s_best = ideal_dist_init_array(s_axis)
    s_best /= np.sum(s_best)

    o_worst = o_axis * 0
    o_worst[0] = 0.5
    o_worst[np.argmin(np.abs(
        o_axis - ((np.mean(opinion[opinion > 0])) -
                  np.mean(opinion[opinion <= 0]))
    ))] = 0.5

    s_worst = s_axis * 0
    s_worst[0] = 1
    
    # scales
    
    o_scale_worst = o_scale_best = self.div(o_worst, o_best, t_err=self.t_err)
    s_scale_worst = s_scale_best = self.div(s_worst, s_best, t_err=self.t_err)
    if not self.use_js_divergence:
      o_scale_worst = self.div(o_best, o_worst, t_err=self.t_err)
      s_scale_worst = self.div(s_best, s_worst, t_err=self.t_err)

    return {
        self.prefix + '-best-o': self.div(o_pmf, o_best, t_err=self.t_err) / o_scale_best,
        self.prefix + '-best-s': self.div(s_pmf, s_best, t_err=self.t_err) / s_scale_best,
        self.prefix + '-worst-o': self.div(o_pmf, o_worst, t_err=self.t_err) / o_scale_worst,
        self.prefix + '-worst-s': self.div(s_pmf, s_worst, t_err=self.t_err) / s_scale_worst,
    }
