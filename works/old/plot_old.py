
# rest figures


# scatter_data(plt, np.array(vals_grad_index), np.array(vals_0d['bc_hom_smpl']), is_consensus, is_not_consensus, is_near_diag, is_not_near_diag)

assert False


pat_files_raw_op = pat_files_raw[::2]
pat_files_raw_st = pat_files_raw[1::2]

kde_gradation_cache = []

g_index_cache = []

event_cache = []

for r in pat_files_raw_op, pat_files_raw_st:

  gradation, triads, in_degree, \
    d_opinion, p_last, g_index_mean_by_rec_active, \
      event_step_, event_unfollow_, event_follow_, \
        active_steps, smpl_rel, smpl_rel_dis_nw, smpl_rel_n_cc = [
      np.array([x[k] for x in r]) if not k.startswith('event') else [x[k] for x in r if k in x]
      for k in (
        'grad_index', 'triads', 'in_degree', 
        'opinion_diff', 'p_last', 'g_index_mean_active',
        'event_step', 'event_unfollow', 'event_follow',
        'active_step', 'smpl_pearson_rel', 'smpl_rec_dis_network', 'smpl_rec_concordant_n',
      )
  ]

  # in_degree_alpha, in_degree_p, in_degree_r = in_degree.T.copy()

  # in_degree_bound = 10
  # in_degree_alpha[in_degree_alpha > in_degree_bound] = in_degree_bound
  # in_degree_alpha[in_degree_r <= 0] = in_degree_bound

  # is_consensus

  is_consensus = p_last < consensus_threshold
  is_not_consensus = np.logical_not(is_consensus)

  # is_near_diag

  rd_rate_vec_all = np.array(
      [rd_rate_vec] * int(len(r) / rd_rate_vec.size)).flatten()
  is_near_diag = np.logical_and(rd_rate_vec_all > -1, rd_rate_vec_all < 1)
  is_not_near_diag = np.logical_not(is_near_diag)

  # gradation - consensus

  d = 0.005
  metrics = np.arange(0.3, 1 + d, d)
  kde_nc_raw = gaussian_kde(gradation[is_not_consensus])(metrics)
  kde_c_raw = gaussian_kde(gradation[is_consensus])(metrics)

  kde_all_raw = kde_nc_raw + kde_c_raw
  kde_all = kde_all_raw / (np.sum(kde_all_raw) * d)
  kde_nc = kde_nc_raw / (np.sum(kde_all_raw) * d)
  kde_c = kde_all - kde_nc
  kde_ratio_c = kde_c / kde_all

  kde_gradation_cache.append([metrics, kde_all, kde_nc, kde_c, kde_ratio_c])

  print(
      'gradation',
      np.mean(gradation[is_consensus]),
      np.mean(gradation[is_not_consensus])
  )

  # env index
  # gradation - triads

  g_index_cache.append([
      gradation,
      is_consensus, is_not_consensus,
      is_near_diag, is_not_near_diag,
      triads, 
      g_index_mean_by_rec_active,
      smpl_rel,
      smpl_rel_dis_nw,
      smpl_rel_n_cc,
  ])

  d_opinion[d_opinion < 0] = 0
  
  print(
      'd_opinion',
      np.mean(d_opinion[is_consensus]),
      np.mean(d_opinion[is_not_consensus])
  )
  
  # event
  event_step = [decompress_b64_to_array(x, int) for x in event_step_]
  event_unfollow = [decompress_b64_to_array(x, float) for x in event_unfollow_]
  event_follow = [decompress_b64_to_array(x, float) for x in event_follow_]
  
  event_step_normalized = [
    x / active_steps[i] for i, x in enumerate(event_step)
  ]
  
  event_cache.append((
    event_step, event_step_normalized, event_unfollow, event_follow
  ))

event_cache_op, event_cache_st = event_cache
event_flattened_op = [[], [], [], []]
event_flattened_st = [[], [], [], []]
for i in range(4):
  for e in event_cache_op[i]: event_flattened_op[i] += list(e)
  for e in event_cache_st[i]: event_flattened_st[i] += list(e)
# _x = np.arange(0, 1.2, 0.01)
# eopd = gaussian_kde(eop)(_x)
# estd = gaussian_kde(est)(_x)
# plt.plot(_x, eopd)
# plt.plot(_x, estd)


# triads



(g_op, c_op, nc_op, d_op, nd_op, tr_op, gi_op, s_rec_op, s_rec_nw_op, s_rec_cc_op), \
    (g_st, c_st, nc_st, d_st, nd_st, tr_st, gi_st, s_rec_st, s_rec_nw_st, s_rec_cc_st) = g_index_cache



# env index

fig, (axfreq, axst2, axop2) = plt_figure(n_col=3, hw_ratio=4/5, total_width=18)

kde_cl_op_ = gaussian_kde(gi_op)
kde_cl_st_ = gaussian_kde(gi_st)

metrics = np.linspace(0.2, 1, 200)
kde_cl_op = kde_cl_op_(metrics)
kde_cl_st = kde_cl_st_(metrics)

axfreq.plot(metrics, kde_cl_st, label='structure')
axfreq.plot(metrics, kde_cl_op, label='opinion')
axfreq.legend()

gi_st_stat = scatter_data(axst2, g_st, gi_st, c_st, nc_st, d_st, nd_st)
gi_op_stat = scatter_data(axop2, g_op, gi_op, c_op,
                          nc_op, d_op, nd_op, legend=True)

gi_stat_diff = gi_op_stat - gi_st_stat

axfreq.set_title('(a) PDF of environment index', loc='left')
axst2.set_title('(b) structure', loc='left')
axop2.set_title('(c) opinion', loc='left')

axfreq.set_xlabel('environment index')
axfreq.set_ylabel('probability')

for _ in (axst2, axop2):
  _.set_ylabel('environment index')
  _.set_xlabel('gradation index')
  _.set_ylim(0.2, 1)

plt.tight_layout()
show_fig('grad_env_index_rel')

stat_diff_all = np.concatenate([tr_stat_diff, gi_stat_diff[1:]], axis=0)
numpy_to_latex_table(
    stat_diff_all, f'{BASE_PATH}/stat_diff_all.tex',
    row_labels=['grad. index', '\#triads', 'env. index'],
    col_labels=['ND;P', 'ND;C', 'D;P', 'D;C'])

# s_rec?

# s_rec_nw_op_2 = s_rec_nw_op[:, :, 2].astype(float)
# s_rec_nw_op_mask = np.isfinite(s_rec_nw_op_2)
# i = 6
# plt.scatter(g_op[s_rec_nw_op_mask[:, i]], s_rec_nw_op_2[:, i][s_rec_nw_op_mask[:, i]], s=2)

# relation between opinion similarity & network distance

fig, (axfreq, axst3, axop3) = plt_figure(n_col=3, hw_ratio=4/5, total_width=18)

y_op = np.mean(s_rec_op.astype(float), axis=1)
_y_op = np.logical_not(np.isnan(y_op))
scatter_data(axop3, g_op[_y_op], y_op[_y_op], c_op[_y_op], nc_op[_y_op], d_op[_y_op], nd_op[_y_op])

y_st = np.mean(s_rec_st.astype(float), axis=1)
_y_st = np.logical_not(np.isnan(y_st))
scatter_data(axst3, g_st[_y_st], y_st[_y_st], c_st[_y_st], nc_st[_y_st], d_st[_y_st], nd_st[_y_st])

kde_cl_op_ = gaussian_kde(y_op)
kde_cl_st_ = gaussian_kde(y_st)

metrics = np.linspace(-0.1, 1.1, 200)
kde_cl_op = kde_cl_op_(metrics)
kde_cl_st = kde_cl_st_(metrics)

axfreq.plot(metrics, kde_cl_st, label='structure')
axfreq.plot(metrics, kde_cl_op, label='opinion')
axfreq.legend()

for _ in (axst3, axop3):
  _.set_ylabel('pearson rel.')
  _.set_xlabel('gradation index')
  _.set_ylim(-0.1, 1.1)

plt.tight_layout()
show_fig('grad_op_st_sim_rel_rel')


fig, (axfreq, axst4, axop4) = plt_figure(n_col=3, hw_ratio=4/5, total_width=18)

y_op = np.mean(s_rec_nw_op.astype(float)[:, :, 2], axis=1)
_y_op = np.logical_not(np.isnan(y_op))
scatter_data(axop4, g_op[_y_op], y_op[_y_op], c_op[_y_op], nc_op[_y_op], d_op[_y_op], nd_op[_y_op])

y_st = np.mean(s_rec_nw_st.astype(float)[:, :, 2], axis=1)
_y_st = np.logical_not(np.isnan(y_st))
scatter_data(axst4, g_st[_y_st], y_st[_y_st], c_st[_y_st], nc_st[_y_st], d_st[_y_st], nd_st[_y_st])

kde_cl_op_ = gaussian_kde(y_op[_y_op])
kde_cl_st_ = gaussian_kde(y_st[_y_st])

metrics = np.linspace(1.6, 2.1, 200)
kde_cl_op = kde_cl_op_(metrics)
kde_cl_st = kde_cl_st_(metrics)

axfreq.plot(metrics, kde_cl_st, label='structure')
axfreq.plot(metrics, kde_cl_op, label='opinion')
axfreq.legend()

for _ in (axst4, axop4):
  _.set_ylabel('pearson rel.')
  _.set_xlabel('gradation index')
  _.set_ylim(1.65, 2.05)

plt.tight_layout()
show_fig('grad_op_st_rec_rel_rel')

# event freq and time quant

keys = ['grad_index', 'event_count', 'event_step_mean', 'active_step']
mats = []
for k in keys:
  vals = vals_0d[k]
  vals = np.array(vals).reshape((-1, 8, 8, 2))
  vals_op = vals[..., ::2]
  vals_st = vals[..., 1::2]
  mats.append((vals_op, vals_st))
  
(giop, gist), (ecop, ecst), (esmop, esmst), (asop, asst) = mats