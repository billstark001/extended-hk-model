import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

# drafts


# data_ = [x for x in data[9::9] if x['p_last'] >= -0.6]
# plt.scatter([x['steepness']for x in data_], [x['active_step'] for x in data_], s=2)
# plt.yscale('log')
# plt.xscale('log')
# plt.grid(True)
# plt.show()

data_ = [x for x in data[::9] if x['p_last'] >= 0.6]
data_x = [x['steepness']for x in data_]
data_y = [x['active_step'] for x in data_]

X = np.array(data_x)
X[X < 0.125] = 0.125
Y = np.array(data_y)

X = np.log10(X)
Y = np.log10(Y)

kde_X = gaussian_kde(X)
kde_Y = gaussian_kde(Y)

kde_XY = gaussian_kde(np.vstack([X, Y]))

grid_resolution = 400
x_grid = np.linspace(X.min(), X.max(), grid_resolution)
y_grid = np.linspace(Y.min(), Y.max(), grid_resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

pdf_X = np.reshape(kde_X(x_grid), x_grid.shape)
pdf_Y = np.reshape(kde_Y(y_grid), y_grid.shape)
pdf_XY = np.reshape(kde_XY(np.vstack([X_grid.ravel(), Y_grid.ravel()])), X_grid.shape)

mutual_info = np.sum(pdf_XY * np.log(pdf_XY / (pdf_X[:, None] * pdf_Y[None, :])))
mutual_info


keys = ['grad_index', 'event_count', 'event_step_mean', 'active_step']
mats = []
for k in keys:
  vals = [x[k] for x in pat_stats_set]
  vals = np.array(vals).reshape((-1, 14))
  vals_op = vals[:, ::2]
  vals_st = vals[:, 1::2]
  mats.append((vals_op, vals_st))
  
(giop, gist), (ecop, ecst), (esmop, esmst), (asop, asst) = mats
