from typing import List

import numpy as np
from scipy.stats import gaussian_kde

from utils.file import read_records

from works.plot.visuals import plt_figure, plt_save_and_close

import works.config as cfg


# Category labels for plots
cat_labels = ['P;P', 'P;C', 'H;P', 'H;C']


def load_pattern_data(pat_file_paths: List[str]):
  """Load pattern data from json files"""
  full_sim_len = cfg.rewiring_rate_array.shape[0] * \
      cfg.decay_rate_array.shape[0] * len(cfg.rs_names)

  vals_0d, vals_non_0d = read_records(pat_file_paths, full_sim_len)

  # Process loaded data
  vals_0d['mean_vars_smpl'] = np.mean(vals_non_0d['mean_vars_smpl'], axis=1)
  vals_0d['bc_hom_smpl'] = np.mean(vals_non_0d['bc_hom_smpl'], axis=1)

  return vals_0d, vals_non_0d


def prepare_heatmap_data(vals_0d):
  """Prepare data for heatmaps"""
  fields_to_draw_heatmap = ['active_step',
                            'grad_index', 'p_last', 'triads', 'mean_vars_smpl']
  pat_csv_values_ = [vals_0d[k].to_numpy(
      dtype=float) for k in fields_to_draw_heatmap]
  pat_csv_values_raw = np.array(pat_csv_values_)

  pat_csv_values = pat_csv_values_raw.reshape((
      len(fields_to_draw_heatmap),
      -1,
      cfg.rewiring_rate_array.shape[0],
      cfg.decay_rate_array.shape[0],
      len(cfg.rs_names),
  ))

  # axes: (#sim, rewiring, decay, recsys)
  active_steps, grad_indices, hs_last, triads, mean_vars = pat_csv_values

  # Calculate averages across simulations
  m_active_step = np.mean(active_steps, axis=0, dtype=float)
  m_grad_index = np.mean(grad_indices, axis=0, dtype=float)

  consensus_threshold = 0.6
  m_hs_last = np.mean(hs_last, axis=0, dtype=float)
  m_is_consensus = np.mean(hs_last.astype(
      float) < consensus_threshold, axis=0, dtype=float)

  m_active_step_op = np.log10(m_active_step[..., 0])
  m_active_step_st = np.log10(m_active_step[..., 1])

  m_grad_index_op = m_grad_index[..., 0]
  m_grad_index_st = m_grad_index[..., 1]

  mean_triads = np.mean(triads, axis=0)
  mean_triads_op = np.log10(mean_triads[..., 0])
  mean_triads_st = np.log10(mean_triads[..., 1])

  results = {
      'active_step_op': m_active_step_op,
      'active_step_st': m_active_step_st,
      'grad_index_op': m_grad_index_op,
      'grad_index_st': m_grad_index_st,
      'triads_op': mean_triads_op,
      'triads_st': mean_triads_st,
      'consensus_threshold': consensus_threshold
  }

  return results


def prepare_vectorized_data(vals_0d, consensus_threshold):
  """Prepare vectorized data for scatter plots"""
  # Prepare rewiring/decay matrix
  rewiring_mat = np.repeat(cfg.rewiring_rate_array.reshape(
      (-1, 1)), axis=1, repeats=cfg.decay_rate_array.size)
  decay_mat = np.repeat(cfg.decay_rate_array.reshape(
      (1, -1)), axis=0, repeats=cfg.rewiring_rate_array.size)
  rd_rate_mat = np.log10(rewiring_mat) - np.log10(decay_mat)

  # Get flat vectors
  rd_rate_vec = rd_rate_mat.reshape(-1)

  # Consensus flags
  is_consensus = np.array(vals_0d['p_last'] < consensus_threshold)
  is_not_consensus = np.logical_not(is_consensus)

  # Near diagonal flags
  rd_rate_vec_2 = np.vstack([[rd_rate_vec], [rd_rate_vec]]).T.flatten()
  rd_rate_vec_all = np.array(
      [rd_rate_vec_2] * int(len(is_consensus) / rd_rate_vec_2.size)).flatten()
  is_near_diag = np.logical_and(rd_rate_vec_all > -1, rd_rate_vec_all < 1)
  is_not_near_diag = np.logical_not(is_near_diag)

  results = {
      'rewiring_mat': rewiring_mat,
      'decay_mat': decay_mat,
      'rd_rate_mat': rd_rate_mat,
      'rd_rate_vec': rd_rate_vec,
      'rd_rate_vec_all': rd_rate_vec_all,
      'is_consensus': is_consensus,
      'is_not_consensus': is_not_consensus,
      'is_near_diag': is_near_diag,
      'is_not_near_diag': is_not_near_diag,
  }

  return results


def plot_grad_index_distribution(vals_grad_index, is_consensus, is_not_consensus, save_path):
  """Plot distributions of gradation index by consensus"""
  d = 0.005
  metrics = np.arange(0.4, 1 + d, d)
  gradation_all = vals_grad_index

  kde_nc_raw = gaussian_kde(gradation_all[is_not_consensus])(metrics)
  kde_c_raw = gaussian_kde(gradation_all[is_consensus])(metrics)

  kde_all_raw = kde_nc_raw + kde_c_raw
  kde_all = kde_all_raw / (np.sum(kde_all_raw) * d)
  kde_nc = kde_nc_raw / (np.sum(kde_all_raw) * d)
  kde_c = kde_all - kde_nc
  kde_ratio_c = kde_c / kde_all

  fig, (ax_grad, ax_ratio) = plt_figure(n_col=2, total_width=11)

  ax_grad.plot(metrics, kde_nc, label='polarized', color='tab:red')
  ax_grad.plot(metrics, kde_c, label='consensual', color='tab:green')
  ax_grad.plot(metrics, kde_all, label='all', color='tab:blue')
  ax_grad.legend()

  ax_ratio.plot(metrics, kde_ratio_c)

  ax_grad.set_title('(a) dist. of gradation index', loc='left')
  ax_ratio.set_title('(c) %consensual cases', loc='left')

  ax_grad.set_yticks(np.array([0, 1, 2, 4, 7, 12]))
  for _ in (ax_grad, ax_ratio):
    _.set_xlabel('gradation index')

  ax_grad.set_ylabel('prob. density')
  ax_ratio.set_ylabel('ratio')

  plt_save_and_close(save_path)
