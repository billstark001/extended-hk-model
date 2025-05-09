from typing import Dict, Union, Optional, Tuple
from numpy.typing import NDArray

import numpy as np
import networkx as nx
from scipy.stats import norm, gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad


def ideal_dist_init_array(x: np.ndarray, k=2):
  ret_array = x / -k + 1
  ret_array[x > k] = 0
  ret_array[x < 0] = 0
  return ret_array


def kl_divergence_continuous(p_func, q_func, xmin=0, xmax=2, t_err=1e-13) -> float:
  def integrand(x):
    p = max(p_func(x), t_err)
    q = max(q_func(x), t_err)
    return p * (np.log(p) - np.log(q))
  val, _ = quad(integrand, xmin, xmax, limit=100)
  return val


def js_divergence_continuous(p_func, q_func, xmin=0, xmax=2, t_err=1e-13) -> float:
  def integrand(x):
    p = max(p_func(x), t_err)
    q = max(q_func(x), t_err)
    m = (p + q) / 2
    return 0.5 * (p * (np.log(p) - np.log(m)) + q * (np.log(q) - np.log(m)))
  val, _ = quad(integrand, xmin, xmax, limit=100)
  return np.sqrt(val)


LINSPACE_SMPL_COUNT = 128

def kl_divergence_continuous_fast(p_func, q_func, xmin=0, xmax=2, n: NDArray | int = LINSPACE_SMPL_COUNT, t_err=1e-13):
  if isinstance(n, np.ndarray):
    xs = n
    p_vals = np.maximum(p_func, t_err)
    q_vals = np.maximum(q_func, t_err)
  else:
    xs = np.linspace(xmin, xmax, n)
    p_vals = np.maximum(p_func(xs), t_err)
    q_vals = np.maximum(q_func(xs), t_err)
  integrand = p_vals * (np.log(p_vals) - np.log(q_vals))
  return np.trapz(integrand, xs)


def fast_trapz(y: NDArray, x: NDArray):
  h = x[1] - x[0]
  result = h * (y.sum() - 0.5 * (y[0] + y[-1]))
  return result

def js_divergence_continuous_fast(p_func, q_func, xmin=0, xmax=2, n: NDArray | int = LINSPACE_SMPL_COUNT, t_err=1e-13):
  if isinstance(n, np.ndarray):
    xs = n
    p_vals = np.maximum(p_func, t_err)
    q_vals = np.maximum(q_func, t_err)
  else:
    xs = np.linspace(xmin, xmax, n)
    p_vals = np.maximum(p_func(xs), t_err)
    q_vals = np.maximum(q_func(xs), t_err)
  m_vals = 0.5 * (p_vals + q_vals)
  integrand = 0.5 * (p_vals * (np.log(p_vals) - np.log(m_vals)) +
                     q_vals * (np.log(q_vals) - np.log(m_vals)))
  val = fast_trapz(integrand, xs)
  # val = np.trapz(integrand, xs)
  return np.sqrt(val)


def kde_min_bw_factory(min_bandwidth):
  def min_bw_factor(kde_obj):
    default_factor = kde_obj.scotts_factor()
    min_factor = min_bandwidth / np.std(kde_obj.dataset, ddof=1)
    return max(default_factor, min_factor)
  return min_bw_factor

def kde_min_bw_calc(data: NDArray, min_bw=0.1):
  std = np.std(data, ddof=1)
  default_bw = 1.06 * std * len(data) ** (-1 / 5)  # 例如scott
  bw = max(default_bw, min_bw)
  return bw

def min_bandwidth_enforcer(bandwidth, min_bandwidth, data):
  return max(bandwidth, min_bandwidth, np.std(data, ddof=1)*0.1)  # 举例

def get_kde_pdf(data, min_bandwidth: float, xmin: float, xmax: float):
  # 如果样本全为某个点，KDE会报错，这里做特殊处理
  if np.all(data == data[0]):
    # 返回一个峰值在这个点的近似delta分布
    def delta_like_pdf(x):
      return norm.pdf(x, data[0], min_bandwidth)
      # return 1.0 if np.isclose(x[0], data[0]) else 0.0
    return delta_like_pdf

  bw_method = kde_min_bw_factory(min_bandwidth)
  data_smpl = data # np.random.choice(data, size=2000, replace=False)
  kde = gaussian_kde(data_smpl, bw_method=bw_method)
  return kde

  # kde = KernelDensity(
  #   kernel='gaussian', 
  #   bandwidth=kde_min_bw_calc(data, min_bw=min_bandwidth))
  # kde.fit(data[:, None])  # 需要二维 shape=(n_samples, n_features)
  
  # def ret(x: NDArray):
  #   log_density = kde.score_samples(x[:, None])    # 返回log密度
  #   density = np.exp(log_density)         # 取指数得到真正的密度
  #   return density
  
  # return ret


class DistanceCollectorContinuous:
  def __init__(
      self,
      min_bandwidth: Optional[float] = 0.05,
      node_cnt = 500,
      t_err: float = 1e-10,
      use_js_divergence: bool = False,
      use_debug_data: bool = False,
      t_opinion: float = 0.4,
      k: float = 2.0
  ):
    self.min_bandwidth = min_bandwidth
    self.t_err = t_err
    self.use_js_divergence = use_js_divergence
    self.use_debug_data = use_debug_data
    self.t_opinion = t_opinion
    self.k = k
    self.div = js_divergence_continuous_fast \
        if use_js_divergence else kl_divergence_continuous_fast
        
    
    err_range = self.min_bandwidth * 4
    self.axis = np.linspace(0 - err_range, k + err_range, LINSPACE_SMPL_COUNT)
    
    
    o_rand_vals = ideal_dist_init_array(self.axis, k=k)
    if o_rand_vals.sum() > 0:
      o_rand_vals /= np.trapz(o_rand_vals, self.axis)
    else:
      o_rand_vals[:] = 1.0 / (k - 0)
    
    self.o_rand_vals = o_rand_vals
    self.triu_indices = np.triu_indices(node_cnt, k=1)

  def collect(
      self,
      prefix: str,
      digraph: nx.DiGraph, opinion: np.ndarray,
      *args, **kwargs,
  ) -> Union[float, np.ndarray, Dict[str, Union[float, np.ndarray]]]:

    k = self.k
    # sampling
    o_slice_mat = np.tile(opinion.reshape((opinion.size, 1)), opinion.size)
    o_sample_mat = np.abs(o_slice_mat - o_slice_mat.T)
    o_sample = o_sample_mat[self.triu_indices]
    o_sample = np.clip(o_sample, 0, k)
    neighbors = np.array(digraph.edges)
    s_sample = np.abs(opinion[neighbors[:, 1]] - opinion[neighbors[:, 0]])
    s_sample = np.clip(s_sample, 0, k)

    # KDE for pmf
    o_pdf = get_kde_pdf(o_sample, self.min_bandwidth, 0, k)
    s_pdf = get_kde_pdf(s_sample, self.min_bandwidth, 0, k)


    # 构造理想分布，用于归一化/对比
    err_range = self.min_bandwidth * 4
    axis = self.axis
    s_rand_vals = o_rand_vals = self.o_rand_vals

    # def o_rand_pdf(x):
    #   return np.interp(x, axis, o_rand_vals)
    # s_rand_pdf = o_rand_pdf

    # worst cases
    o_worst_b = o_sample[o_sample >= self.t_opinion]
    o_worst_v = self.t_opinion if o_worst_b.size == 0 else np.mean(o_worst_b)

    o_worst_vals = \
        0.5 * norm.pdf(axis, 0, self.min_bandwidth) + \
        0.5 * norm.pdf(axis, o_worst_v, self.min_bandwidth)
    if o_worst_vals.sum() > 0:
      o_worst_vals /= fast_trapz(o_worst_vals, axis)

    # def o_worst_pdf(x):
    #   return np.interp(x, axis, o_worst_vals)

    s_worst_vals = norm.pdf(axis, 0, self.min_bandwidth)
    if s_worst_vals.sum() > 0:
      s_worst_vals /= fast_trapz(s_worst_vals, axis)

    # def s_worst_pdf(x):
    #   return np.interp(x, axis, s_worst_vals)

    # scales

    params_dict = dict(
        xmin=0 - err_range, xmax=k + err_range, t_err=self.t_err,
        n=axis,
    )
    o_scale_worst = o_scale_rand = self.div(
        o_worst_vals, o_rand_vals,
        **params_dict
    )
    s_scale_worst = s_scale_rand = self.div(
        s_worst_vals, s_rand_vals,
        **params_dict
    )
    if not self.use_js_divergence:
      o_scale_worst = self.div(
          o_rand_vals, o_worst_vals,
          **params_dict
      )
      s_scale_worst = self.div(
          s_rand_vals, s_worst_vals,
          **params_dict
      )

    o_vals = o_pdf(axis)
    s_vals = s_pdf(axis)

    debug_data = {}
    if self.use_debug_data:
      debug_data = {
          prefix + '-kde-o-sample': o_sample,
          prefix + '-kde-s-sample': s_sample,
          prefix + '-axis': axis,
          prefix + '-rand-o-pdf': o_rand_vals,
          prefix + '-rand-s-pdf': o_rand_vals,
          prefix + '-worst-o-pdf': o_worst_vals,
          prefix + '-worst-s-pdf': s_worst_vals,
      }

    return {
        prefix + '-rand-o': self.div(o_vals, o_rand_vals,  **params_dict) / o_scale_rand,
        prefix + '-rand-s': self.div(s_vals, s_rand_vals,  **params_dict) / s_scale_rand,
        prefix + '-worst-o': self.div(o_vals, o_worst_vals,  **params_dict) / o_scale_worst,
        prefix + '-worst-s': self.div(s_vals, s_worst_vals,  **params_dict) / s_scale_worst,
        **debug_data,
    }
